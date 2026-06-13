import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

import reservoirpy as rpy
from reservoirpy.hyper import parallel_research, plot_hyperopt_report
from reservoirpy.nodes import Reservoir, Ridge, ScikitLearnNode

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())
from utils import align_fmri_to_movie_window, load_movie_timing

# TO EDIT THE CODE FOR YOUR NEEDS
## TODO: to load your data (input / outputs) go to the placeholder "ZONE TO EDIT"

# Choose/change subject index here:
SUBJECT_INDEX = 4
SPLIT_FRACTION = 0.85

# Extra reservoir training parameters
# WARMUP = 40 #20 # number of initial time steps ignored during reservoir training

# BOLD fMRI parameter vs. behavioral output (emotion valence)
DEFAULT_LAG_SECONDS = 5.0

# Hyper-parameter (HP) search parameters
## If want to do shorter/longer hyper-parameter searches
## then, reduce/augment one or more of these variables
## - hp_max_evals: number of set of HP explored
## - N: number of neurons inside the reservoir (= computational power)
## - instances_per_trial: number of reservoir trained instances to average the results/loss over
HYPER_SEARCH_CONFIG = {
    "exp": "brainhack",
    "hp_max_evals": 200, # -> use 200+ for more broader parameter exploration
    "hp_method": "random", # -> use "tpe" for bayesian optimization but without parellelization
    "seed": 42,
    "instances_per_trial": 5, # -> use 5 for more robustness
    "hp_space": {
        "N": ["choice", 2000], # -> use 1000 or 2000 for more computational power
        "sr": ["loguniform", 1e-3, 3],
        "lr": ["loguniform", 1e-4, 1],
        "input_scaling": ["loguniform", 1e-5, 3], #"input_scaling": ["choice", 1.0],
        "ridge": ["loguniform", 1e-8, 1e5],
        "seed": ["choice", 1234],
        "warmup": ["quniform", 0, 50, 5],
        "lag_seconds": ["uniform", 0.0, 15.0],
    }
}



# Parellization of subject processing?
## /!\ Parellalisation means more RAM usage
## (number of workers used by ReservoirPy when fitting each ESN)
RESERVOIR_WORKERS = 1 # NO Parallization
#RESERVOIR_WORKERS = -1 # "-1" means all CPU cores are used


#---
#---
#---



# Hyperopt output folder
HYPEROPT_SEARCH_DIR = Path("../hyperopt-search")
# Plot parameters
HYPEROPT_PLOT_FIGSIZE = (24, 16) # (horizontal axis, vertical axis)
HYPEROPT_PLOT_DPI = 100
HYPEROPT_REPORT_PREFIX = "hyper-search-report"
ALIGNMENT_DIAGNOSTIC_FEATURE_INDEX = 0
SHOW_FINAL_PREDICTION_PLOT = True


# Details not useful to humans
RESULT_CALL_RE = re.compile(r"^(?P<prefix>.+?)_(?:\d+call|call\d+)(?P<suffix>\.json)$")
# 
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.base",
)

def train_and_evaluate(params, dataset, warmup):
    esn = build_esn(**params)

    esn.fit(
        dataset["X_train_scaled"],
        dataset["Y_train_scaled"],
        warmup=int(warmup),
        workers=dataset["workers"],
    )

    Y_pred = predict_test_after_train_replay(
        esn,
        dataset["X_train_scaled"],
        dataset["X_test_scaled"],
        dataset["y_scaler"],
    )

    return compute_metrics(dataset["Y_true_flat"], Y_pred)



def ask_yes_no(prompt, default=False):
    """Ask an interactive yes/no question, with a safe default for scripts."""
    yes = {"y", "yes"}
    no = {"n", "no"}

    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            print("\nNo interactive input detected; hyperopt plots disabled.")
            return default

        if answer == "":
            return default
        if answer in yes:
            return True
        if answer in no:
            return False

        print("Please answer with yes or no.")


def index_to_letter_suffix(index):
    suffix = ""
    while True:
        index, remainder = divmod(index, 26)
        suffix = chr(ord("a") + remainder) + suffix
        index -= 1
        if index < 0:
            return suffix


def timestamped_run_name(exp_name, now=None):
    now = now or datetime.now()
    minute_stamp = now.strftime("%Y-%m-%d_%Hh%M")
    run_name_base = f"{exp_name}_{minute_stamp}"
    HYPEROPT_SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    suffix_index = 0
    while True:
        run_name = f"{run_name_base}{index_to_letter_suffix(suffix_index)}"
        if not (HYPEROPT_SEARCH_DIR / run_name).exists():
            return run_name
        suffix_index += 1


def prepare_hyperopt_run(base_config):
    base_exp = base_config["exp"]
    run_name = timestamped_run_name(base_exp)
    experiment_dir = HYPEROPT_SEARCH_DIR / run_name
    experiment_dir.mkdir(parents=True, exist_ok=False)

    run_config = dict(base_config)
    run_config["base_exp"] = base_exp
    run_config["exp"] = run_name

    return run_config, experiment_dir


def hyperopt_result_loss_label(result_filename):
    match = RESULT_CALL_RE.match(result_filename)
    prefix = match.group("prefix") if match is not None else Path(result_filename).stem
    if prefix.endswith("_results"):
        prefix = prefix[:-len("_results")]
    return f"loss{prefix}"


def move_hyperopt_results_to_run_dir(experiment_dir):
    generated_results_dir = experiment_dir / "results"
    if not generated_results_dir.exists():
        return 0, []

    moved_count = 0
    example_files = []
    result_files = sorted(
        [path for path in generated_results_dir.glob("*.json") if path.is_file()],
        key=lambda path: (path.stat().st_mtime, path.name),
    )

    for trial_number, result_file in enumerate(result_files, start=1):
        loss_label = hyperopt_result_loss_label(result_file.name)
        target_file = experiment_dir / f"{loss_label}_trial{trial_number:04d}.json"
        result_file.rename(target_file)

        moved_count += 1
        if len(example_files) < 3:
            example_files.append(target_file.name)

    if generated_results_dir.exists() and not any(generated_results_dir.iterdir()):
        generated_results_dir.rmdir()

    return moved_count, example_files


def build_flat_hyperopt_report_dir(experiment_dir, flat_root):
    flat_exp_dir = flat_root / experiment_dir.name
    flat_results_dir = flat_exp_dir / "results"
    flat_results_dir.mkdir(parents=True, exist_ok=True)

    source_results_dir = experiment_dir / "results"
    if source_results_dir.exists():
        for source_file in sorted(source_results_dir.glob("*.json")):
            if source_file.is_file():
                (flat_results_dir / source_file.name).write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")

    for source_file in sorted(experiment_dir.glob("loss*_trial*.json")):
        if source_file.is_file():
            (flat_results_dir / source_file.name).write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")

    return flat_exp_dir


def as_column_vector(series):
    """Represent a time-varying univariate target as (n_timepoints, 1)."""
    return np.asarray(series, dtype=float).reshape(-1, 1)


def lag_label(lag_seconds):
    return f"lag{float(lag_seconds):.3f}".replace(".", "p")


def ridgecv_alphas_around(best_ridge, decades=2, n_alphas=13):
    best_ridge = max(float(best_ridge), np.finfo(float).tiny)
    return np.logspace(
        np.log10(best_ridge) - decades,
        np.log10(best_ridge) + decades,
        n_alphas,
    )


def prepare_lagged_subject_dataset(
    raw_subject_series,
    event_timing_file,
    movie_behavior,
    *,
    lag_seconds,
    split_fraction,
    workers,
):
    x_aligned = align_fmri_to_movie_window(
        raw_subject_series,
        event_timing_file,
        len(movie_behavior),
        lag_seconds=float(lag_seconds),
    )
    y_aligned = as_column_vector(movie_behavior)

    split_index = int(split_fraction * x_aligned.shape[0])
    X = [x_aligned]
    Y = [y_aligned]

    X_train = [s[:split_index] for s in X]
    Y_train = [s[:split_index] for s in Y]
    X_test = [s[split_index:] for s in X]
    Y_test = [s[split_index:] for s in Y]

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(np.vstack(X_train))

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler.fit(np.vstack(Y_train))

    return {
        "X": X,
        "Y": Y,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "X_train_scaled": [x_scaler.transform(x) for x in X_train],
        "Y_train_scaled": [y_scaler.transform(y) for y in Y_train],
        "X_test_scaled": [x_scaler.transform(x) for x in X_test],
        "Y_true_flat": np.vstack(Y_test),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "split_index": split_index,
        "lag_seconds": float(lag_seconds),
        "workers": workers,
    }


def zscore_1d(series):
    series = np.asarray(series, dtype=float).ravel()
    mean = np.nanmean(series)
    std = np.nanstd(series)
    if not np.isfinite(std) or std == 0:
        return series - mean
    return (series - mean) / std


# def behavior_persistence_baseline(y_series, split_index):
#     """Predict each test behavior sample from the previous aligned behavior sample."""
#     if split_index <= 0:
#         raise ValueError("split_index must be > 0 to build a persistence baseline.")
#
#     y_test = y_series[split_index:]
#     baseline = np.empty_like(y_test)
#     baseline[0] = y_series[split_index - 1]
#     if len(y_test) > 1:
#         baseline[1:] = y_series[split_index:-1]
#     return baseline


def plot_alignment_diagnostic(
    raw_fmri_series,
    aligned_fmri_series,
    movie_behavior,
    event_timing_path,
    lag_seconds,
    target_column,
    feature_index,
    output_path,
):
    movie_onset, movie_duration, run_end = load_movie_timing(event_timing_path)

    raw_fmri_time = np.linspace(0.0, run_end, num=len(raw_fmri_series), endpoint=True)
    behavior_movie_time = np.linspace(
        0.0,
        movie_duration,
        num=len(movie_behavior),
        endpoint=False,
    )
    behavior_run_time = movie_onset + behavior_movie_time
    aligned_movie_time = np.linspace(
        0.0,
        movie_duration,
        num=len(aligned_fmri_series),
        endpoint=False,
    )

    behavior_for_aligned_grid = np.interp(
        aligned_movie_time,
        behavior_movie_time,
        movie_behavior,
    )

    feature_raw = raw_fmri_series[:, feature_index]
    feature_aligned = aligned_fmri_series[:, feature_index]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(
        raw_fmri_time,
        zscore_1d(feature_raw),
        linewidth=1.2,
        alpha=0.9,
        label=f"raw fMRI feature {feature_index}",
    )
    axes[0].plot(
        behavior_run_time,
        zscore_1d(movie_behavior),
        linewidth=1.8,
        alpha=0.85,
        label=f"raw behavior {target_column}",
    )
    axes[0].axvspan(
        movie_onset,
        movie_onset + movie_duration,
        color="0.9",
        alpha=0.35,
        label="movie window",
    )
    axes[0].set_title("Before alignment: raw run time")
    axes[0].set_xlabel("run time (s)")
    axes[0].set_ylabel("z-score")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        aligned_movie_time,
        zscore_1d(feature_aligned),
        linewidth=1.4,
        alpha=0.9,
        label=f"fMRI feature {feature_index} sampled at behavior time + {lag_seconds:g}s",
    )
    axes[1].plot(
        aligned_movie_time,
        zscore_1d(behavior_for_aligned_grid),
        linewidth=1.8,
        alpha=0.85,
        label=f"aligned behavior {target_column}",
    )
    axes[1].set_title("After alignment and hemodynamic lag")
    axes[1].set_xlabel("movie time (s)")
    axes[1].set_ylabel("z-score")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"Alignment diagnostic for {event_timing_path.stem}",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_full_series_prediction(
    y_true,
    y_pred,
    split_index,
    target_column,
    output_path,
    warmup=None,
    show=False,
):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Expected y_true and y_pred to have the same shape, got {y_true.shape} and {y_pred.shape}."
        )

    time_index = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_index, y_true, color="black", linewidth=2.0, label=f"expected {target_column}")
    ax.plot(time_index, y_pred, color="tab:blue", linewidth=1.5, alpha=0.85, label="prediction")

    ax.axvline(
        split_index,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label="train/test split",
    )
    ax.axvspan(0, split_index, color="tab:green", alpha=0.06, label="train")
    ax.axvspan(split_index, len(y_true) - 1, color="tab:red", alpha=0.05, label="test")

    if warmup is not None and warmup > 0:
        ax.axvspan(0, min(warmup, len(y_true) - 1), color="0.5", alpha=0.08, label="warmup")

    ax.set_title("Full time series prediction after hyperparameter optimization")
    ax.set_xlabel("aligned timepoint")
    ax.set_ylabel(target_column)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def build_esn(*, N, sr, lr, input_scaling, ridge, seed):
    """
    ESN (Echo State Network) is the whole model including 
    - the reservoir (= untrained recurrent neural network)
    - the readout (= the output layer = ridge regression)
    
    """
    reservoir = Reservoir(
        units=int(N),
        sr=float(sr),
        lr=float(lr),
        input_scaling=float(input_scaling),
        seed=int(seed),
    )
    readout = Ridge(ridge=float(ridge))
    return reservoir >> readout


def ensure_ridgecv_tags_compatibility():
    if hasattr(RidgeCV, "_get_tags"):
        return

    def _get_tags(self):
        return {"multioutput": True}

    RidgeCV._get_tags = _get_tags


def build_final_esn_with_ridgecv(*, N, sr, lr, input_scaling, ridge, seed, alphas):
    ensure_ridgecv_tags_compatibility()

    reservoir = Reservoir(
        units=int(N),
        sr=float(sr),
        lr=float(lr),
        input_scaling=float(input_scaling),
        seed=int(seed),
    )
    readout = ScikitLearnNode(
        model=RidgeCV,
        alphas=alphas,
    )
    return reservoir >> readout, readout


def inverse_transform_predictions(Y_pred_scaled, scaler):
    return [
        scaler.inverse_transform(y_hat)
        for y_hat in Y_pred_scaled
    ]


def predict_independent_series(esn, X_scaled, scaler):
    Y_pred_scaled = []

    for x in X_scaled:
        esn.reset()
        Y_pred_scaled.append(esn.run(x))

    return inverse_transform_predictions(Y_pred_scaled, scaler)


def predict_test_after_train_replay(esn, X_train_scaled, X_test_scaled, scaler):
    if len(X_train_scaled) != len(X_test_scaled):
        raise ValueError(
            f"Expected the same number of train and test series, got {len(X_train_scaled)} and {len(X_test_scaled)}."
        )

    Y_pred_scaled = []
    for x_train, x_test in zip(X_train_scaled, X_test_scaled):
        esn.reset()
        esn.run(x_train)
        Y_pred_scaled.append(esn.run(x_test))

    return inverse_transform_predictions(Y_pred_scaled, scaler)


# def compute_metrics(Y_true_flat, Y_pred):
#     Y_pred_flat = np.vstack(Y_pred)

#     mse_reservoir = mean_squared_error(Y_true_flat, Y_pred_flat)
#     rmse_reservoir = np.sqrt(mse_reservoir)
#     mae_reservoir = mean_absolute_error(Y_true_flat, Y_pred_flat)
#     r2_reservoir = r2_score(Y_true_flat, Y_pred_flat)

#     metrics = {
#         "mse": mse_reservoir,
#         "rmse": rmse_reservoir,
#         "mae": mae_reservoir,
#         "r2": r2_reservoir,
#     }

#     # if Y_baseline_flat is not None:
#     #     mse_baseline = mean_squared_error(Y_true_flat, Y_baseline_flat)
#     #     rmse_baseline = np.sqrt(mse_baseline)
#     #     mae_baseline = mean_absolute_error(Y_true_flat, Y_baseline_flat)
#     #
#     #     metrics.update(
#     #         {
#     #             "mse_baseline": mse_baseline,
#     #             "rmse_baseline": rmse_baseline,
#     #             "mae_baseline": mae_baseline,
#     #             "rmse_ratio": rmse_reservoir / rmse_baseline,
#     #         }
#     #     )

#     return metrics

def compute_metrics(Y_true_flat, Y_pred, Y_baseline_flat=None):
    Y_pred_flat = np.vstack(Y_pred)

    mse_reservoir = mean_squared_error(Y_true_flat, Y_pred_flat)
    rmse_reservoir = np.sqrt(mse_reservoir)
    mae_reservoir = mean_absolute_error(Y_true_flat, Y_pred_flat)
    r2_reservoir = r2_score(Y_true_flat, Y_pred_flat)
    corr_reservoir = float(np.corrcoef(Y_true_flat.ravel(), Y_pred_flat.ravel())[0, 1])

    metrics = {
        "mse": mse_reservoir,
        "rmse": rmse_reservoir,
        "mae": mae_reservoir,
        "r2": r2_reservoir,
        "corr": corr_reservoir,
    }

    # if Y_baseline_flat is not None:
    #     mse_baseline = mean_squared_error(Y_true_flat, Y_baseline_flat)
    #     rmse_baseline = np.sqrt(mse_baseline)
    #     mae_baseline = mean_absolute_error(Y_true_flat, Y_baseline_flat)
    #     corr_baseline = float(np.corrcoef(Y_true_flat.ravel(), Y_baseline_flat.ravel())[0, 1])

    #     metrics.update(
    #         {
    #             "mse_baseline": mse_baseline,
    #             "rmse_baseline": rmse_baseline,
    #             "mae_baseline": mae_baseline,
    #             "rmse_ratio": rmse_reservoir / rmse_baseline,
    #             "mae_ratio": mae_reservoir / mae_baseline,
    #             "corr_baseline": corr_baseline,

    #         }
    #     )

    return metrics



# Define the objective function.
# ReservoirPy hyperopt objectives must take dataset and config first, then
# searched parameters as keyword-only arguments, and return a dict with "loss".
def objective(dataset, config, *, input_scaling, N, sr, lr, ridge, seed, warmup, lag_seconds):
    instances = config["instances_per_trial"]
    variable_seed = int(seed)
    lagged_dataset = prepare_lagged_subject_dataset(
        dataset["raw_subject_series"],
        dataset["event_timing_file"],
        dataset["movie_behavior"],
        lag_seconds=float(lag_seconds),
        split_fraction=dataset["split_fraction"],
        workers=dataset["workers"],
    )

    rmses = []
    maes = []
    r2s = []

    for _ in range(instances):
        params = {
            "N": N,
            "sr": sr,
            "lr": lr,
            "input_scaling": input_scaling,
            "ridge": ridge,
            "seed": variable_seed,
        }

        metrics = train_and_evaluate(params, lagged_dataset, int(warmup))
        rmses.append(metrics["rmse"])
        maes.append(metrics["mae"])
        r2s.append(metrics["r2"])

        variable_seed += 1

    return {
        "loss": float(np.mean(rmses)),
        "rmse": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae": float(np.mean(maes)),
        "r2": float(np.mean(r2s)),
    }


# ---------------
# ---------------
# ---------------
# ---------------
# ---------------
# ---------------
# ---------------
# ---------------

def main():
    import matplotlib.pyplot as plt

    # Reproducibility.
    np.random.seed(42)
    rpy.set_seed(42)

    # show_hyperopt_plots = ask_yes_no(
    #     "Display hyperparameter search plots after optimization? [y/N] ",
    #     default=False,
    # )
    show_hyperopt_plots = False

    #------------------------------------------------------------------
    #------------------------------------------------------------------
    #--- ZONE TO EDIT: UPLOAD YOUR DATASET TO X (inputs) and Y (outputs)
    #------------------------------------------------------------------

    # Choose/change subject index here:
    subject_index = SUBJECT_INDEX


    movie_name = "AfterTheRain"
    target_column = "Throat"
    data_root = SCRIPT_DIR.parent / "data" / "emofilm"
    behav_data = pd.read_csv(data_root / f"Annot_{movie_name}_stim.csv")

    if target_column not in behav_data.columns:
        raise ValueError(f"{target_column!r} not found in behav_data columns: {list(behav_data.columns)}")

    timeseries_archive_path = data_root / f"{movie_name}_schaefer200_parcellated_timeseries.npz"
    with np.load(timeseries_archive_path) as timeseries_archive:
        raw_X = [
            timeseries_archive[f"arr_{i}"]
            for i in range(len(timeseries_archive.files))
        ]

    event_timing_dir = data_root / "event_timings"
    event_timing_files = sorted(event_timing_dir.glob(f"sub-*_{movie_name}_events.tsv"))
    if len(raw_X) != len(event_timing_files):
        raise ValueError(
            f"Timeseries count ({len(raw_X)}) does not match event timing files ({len(event_timing_files)})."
        )

    timing_file = event_timing_files[subject_index]
    raw_subject_series = raw_X[subject_index]

    movie_behavior = (
        behav_data[target_column]
        .astype(float)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy()
    )

    split_fraction = SPLIT_FRACTION #0.7
    preview_dataset = prepare_lagged_subject_dataset(
        raw_subject_series,
        timing_file,
        movie_behavior,
        lag_seconds=DEFAULT_LAG_SECONDS,
        split_fraction=split_fraction,
        workers=RESERVOIR_WORKERS,
    )

    diagnostic_plot_path = data_root / (
        f"{movie_name}_{target_column}_alignment_diagnostic_"
        f"{timing_file.stem}_{lag_label(DEFAULT_LAG_SECONDS)}_"
        f"feature{ALIGNMENT_DIAGNOSTIC_FEATURE_INDEX:03d}.png"
    )
    plot_alignment_diagnostic(
        raw_fmri_series=raw_subject_series,
        aligned_fmri_series=preview_dataset["X"][0],
        movie_behavior=movie_behavior,
        event_timing_path=timing_file,
        lag_seconds=DEFAULT_LAG_SECONDS,
        target_column=target_column,
        feature_index=ALIGNMENT_DIAGNOSTIC_FEATURE_INDEX,
        output_path=diagnostic_plot_path,
    )

    movie_onset, movie_duration, run_end = load_movie_timing(timing_file)
    print(f"Loaded {len(raw_X)} subjects.")
    print(f"Selected subject index: {subject_index}")
    print(f"Selected raw fMRI series: {raw_subject_series.shape}")
    print(f"Movie behavior series: {movie_behavior.shape}")
    print(f"Movie onset/duration/run end: {movie_onset:.3f}s / {movie_duration:.3f}s / {run_end:.3f}s")
    print(f"Preview alignment convention: X(t + {DEFAULT_LAG_SECONDS:g}s) -> Y(t)")
    print(f"Preview aligned fMRI: {preview_dataset['X'][0].shape}")
    print(f"Preview aligned behavior: {preview_dataset['Y'][0].shape}")
    print(f"Alignment diagnostic plot saved to: {diagnostic_plot_path}")




    # ## Create 10 random time series with shape (4000, 200).
    # print("Predicting random data for demo purpose for 10 individuals with 200 voxels on 4,000 time points ...\n")
    # ## NB: Timeseries of differents subjects don't have to be of same length
    # series_list = [
    #     np.random.randn(4000, 200).astype(np.float32)
    #     for _ in range(10)
    # ]

    # # Split train/test by full time series.
    # print("Splitting subjects 8 for training and 2 for testing")
    # train_series = X[:8]
    # test_series = X[8:]

    print(f"Preview training data: {len(preview_dataset['X_train'])} series, each with shape {preview_dataset['X_train'][0].shape}")
    print(f"Preview testing data: {len(preview_dataset['X_test'])} series, each with shape {preview_dataset['X_test'][0].shape}")
    print(f"Preview training target: {len(preview_dataset['Y_train'])} series, each with shape {preview_dataset['Y_train'][0].shape}")
    print(f"Preview testing target: {len(preview_dataset['Y_test'])} series, each with shape {preview_dataset['Y_test'][0].shape}")

    # # Naive behavioral baseline: predict y[t] from the previous aligned y[t-1].
    # Y_baseline = [
    #     behavior_persistence_baseline(y, split_index)
    #     for y in Y
    # ]

    #------------------------------------------------------------------
    #------- END ZONE TO EDIT -----------------------------------------
    #------------------------------------------------------------------

    search_dataset = {
        "raw_subject_series": raw_subject_series,
        "event_timing_file": timing_file,
        "movie_behavior": movie_behavior,
        "split_fraction": split_fraction,
        "workers": RESERVOIR_WORKERS,
    }

    # Concatenate all test series to compute global metrics.
    # Y_baseline_flat = np.vstack(Y_baseline)

    # Configure research.
    hyperopt_config, experiment_dir = prepare_hyperopt_run(HYPER_SEARCH_CONFIG)

    # We precautionously save the configuration in a JSON file.
    config_path = experiment_dir / f"{hyperopt_config['exp']}.config.json"
    with open(config_path, "w+") as f:
        json.dump(hyperopt_config, f, indent=2)

    print()
    print("Hyperparameter search with ReservoirPy parallel_research")
    print(f"Experiment: {hyperopt_config['base_exp']}")
    print(f"Run: {hyperopt_config['exp']}")
    print(f"Evaluations: {hyperopt_config['hp_max_evals']}")
    print(f"Instances per trial: {hyperopt_config['instances_per_trial']}")
    print(f"ReservoirPy fit workers per ESN: {RESERVOIR_WORKERS}")
    print(f"Run folder: {experiment_dir}")
    print(f"Configuration saved to: {config_path}")
    print()

    best_params, best_loss = parallel_research(
        objective,
        search_dataset,
        config_path,
        HYPEROPT_SEARCH_DIR,
    )
    moved_count, result_examples = move_hyperopt_results_to_run_dir(experiment_dir)
    print(f"Results saved in run folder: {experiment_dir}")
    if moved_count:
        print(f"Moved result files into run folder: {moved_count}")
    if result_examples:
        print("Example result paths:", ", ".join(result_examples))
    print()

    best_params = {
        "N": int(best_params["N"]),
        "sr": float(best_params["sr"]),
        "lr": float(best_params["lr"]),
        "input_scaling": float(best_params["input_scaling"]),
        "ridge": float(best_params["ridge"]),
        "seed": int(best_params["seed"]),
        "warmup": int(best_params["warmup"]),
        "lag_seconds": float(best_params["lag_seconds"]),
    }

    print()
    print("Best hyperparameters")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")
    print(f"Best validation RMSE: {best_loss:.6f}")
    print()

    final_dataset = prepare_lagged_subject_dataset(
        raw_subject_series,
        timing_file,
        movie_behavior,
        lag_seconds=best_params["lag_seconds"],
        split_fraction=split_fraction,
        workers=RESERVOIR_WORKERS,
    )

    best_alignment_plot_path = experiment_dir / (
        f"alignment_diagnostic_{timing_file.stem}_"
        f"{lag_label(best_params['lag_seconds'])}_"
        f"feature{ALIGNMENT_DIAGNOSTIC_FEATURE_INDEX:03d}.png"
    )
    plot_alignment_diagnostic(
        raw_fmri_series=raw_subject_series,
        aligned_fmri_series=final_dataset["X"][0],
        movie_behavior=movie_behavior,
        event_timing_path=timing_file,
        lag_seconds=best_params["lag_seconds"],
        target_column=target_column,
        feature_index=ALIGNMENT_DIAGNOSTIC_FEATURE_INDEX,
        output_path=best_alignment_plot_path,
    )

    X = final_dataset["X"]
    Y = final_dataset["Y"]
    X_train_scaled = final_dataset["X_train_scaled"]
    Y_train_scaled = final_dataset["Y_train_scaled"]
    X_full_scaled = [final_dataset["x_scaler"].transform(x) for x in X]
    Y_true_flat = final_dataset["Y_true_flat"]
    y_scaler = final_dataset["y_scaler"]
    split_index = final_dataset["split_index"]

    alphas = ridgecv_alphas_around(best_params["ridge"], decades=2, n_alphas=13)
    final_warmup = best_params["warmup"]
    final_esn_params = {
        "N": best_params["N"],
        "sr": best_params["sr"],
        "lr": best_params["lr"],
        "input_scaling": best_params["input_scaling"],
        "ridge": best_params["ridge"],
        "seed": best_params["seed"],
    }
    final_esn, final_readout = build_final_esn_with_ridgecv(
        **final_esn_params,
        alphas=alphas,
    )
    print(
        "Final RidgeCV alphas tested:",
        np.array2string(alphas, formatter={"float_kind": lambda x: f"{x:.1e}"}),
    )
    print()

    final_esn.fit(
        X_train_scaled,
        Y_train_scaled,
        warmup=final_warmup,
        workers=RESERVOIR_WORKERS,
    )
    full_series_predictions = predict_independent_series(final_esn, X_full_scaled, y_scaler)
    final_predictions = [
        y_pred[split_index:]
        for y_pred in full_series_predictions
    ]
    final_metrics = compute_metrics(Y_true_flat, final_predictions)

    full_train_start = min(final_warmup, split_index)
    full_train_metrics = compute_metrics(
        Y[0][full_train_start:split_index],
        [full_series_predictions[0][full_train_start:split_index]],
    )
    full_series_plot_path = experiment_dir / f"full_series_prediction_{experiment_dir.name}.png"
    plot_full_series_prediction(
        y_true=Y[0],
        y_pred=full_series_predictions[0],
        split_index=split_index,
        target_column=target_column,
        output_path=full_series_plot_path,
        warmup=final_warmup,
        show=SHOW_FINAL_PREDICTION_PLOT,
    )

    # Print results.
    print("Reservoir + RidgeCV after ReservoirPy hyperopt")
    # print("-------------------")
    # print(f"MSE  : {final_metrics['mse']:.6f}")
    print(f"Test RMSE (continuous full-series replay) : {final_metrics['rmse']:.6f}")
    print(f"Test corr (continuous full-series replay) : {final_metrics['corr']:.6f}")
    # print(f"MAE  : {final_metrics['mae']:.6f}")
    print(f"Train RMSE (full-series replay, after warmup) : {full_train_metrics['rmse']:.6f}")
    print(f"Train corr (full-series replay, after warmup) : {full_train_metrics['corr']:.6f}")
    print(f"Best-lag alignment diagnostic plot saved to: {best_alignment_plot_path}")
    print(f"Full-series prediction plot saved to: {full_series_plot_path}")

    # print()
    #
    # # Previous one-step-ahead baseline, not used for the current brain -> behavior task.
    # print("Behavior persistence baseline: y[t] = y[t-1]")
    # # print("------------------------------------")
    # # print(f"MSE  : {final_metrics['mse_baseline']:.6f}")
    # print(f"RMSE : {final_metrics['rmse_baseline']:.6f}")
    # # print(f"MAE  : {final_metrics['mae_baseline']:.6f}")
    #
    # print()
    #
    # print("Ratio reservoir / baseline")
    # # print("--------------------------")
    # # print(f"MSE ratio  : {final_metrics['mse'] / final_metrics['mse_baseline']:.6f}")
    # print(f"RMSE ratio : {final_metrics['rmse_ratio']:.6f}")
    # # print(f"MAE ratio  : {final_metrics['mae'] / final_metrics['mae_baseline']:.6f}")

    # Print the regularization selected during hyperopt and final RidgeCV.
    print()
    print(f"Regularization: Ridge selected during hyperopt objective: {best_params['ridge']:.1e}")
    print(f"Regularization: Alpha selected by final RidgeCV: {final_readout.instances.alpha_:.1e}")

    try:
        import tempfile
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory(prefix="hyperopt-plot-") as plot_dir:
            flat_experiment_dir = build_flat_hyperopt_report_dir(experiment_dir, Path(plot_dir))
            fig = plot_hyperopt_report(
                flat_experiment_dir,
                ("lr", "sr", "input_scaling", "ridge", "warmup", "lag_seconds"),
                metric="r2",
                loss_metric="loss",
                loss_behaviour="min",
                not_log=("warmup", "lag_seconds"),
            )
        fig.set_size_inches(*HYPEROPT_PLOT_FIGSIZE, forward=True)
        fig.set_dpi(HYPEROPT_PLOT_DPI)
        fig.suptitle("ReservoirPy hyperparameter search", y=1.01)

        report_path = experiment_dir / f"{HYPEROPT_REPORT_PREFIX}_{experiment_dir.name}.pdf"
        fig.savefig(report_path, format="pdf", bbox_inches="tight")
        print(f"Hyperopt report saved to: {report_path}")

        if show_hyperopt_plots:
            plt.show()
        else:
            plt.close(fig)
    except Exception as exc:
        print(f"Could not save or display hyperopt plots: {exc}")


if __name__ == "__main__":
    main()
