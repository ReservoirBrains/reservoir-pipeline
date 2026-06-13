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

import sys
sys.path.insert(0, Path("code").resolve().as_posix())
from utils import align_behavior_to_fmri_trs, align_fmri_to_movie_window, slice_fmri_to_movie_window

# TO EDIT THE CODE FOR YOUR NEEDS
## TODO: to load your data (input / outputs) go to the placeholder "ZONE TO EDIT"

# Extra reservoir training parameters
WARMUP = 20 # number of initial time steps ignored during reservoir training

# Hyper-parameter (HP) search parameters
## If want to do shorter/longer hyper-parameter searches
## then, reduce/augment one or more of these variables
## - hp_max_evals: number of set of HP explored
## - N: number of neurons inside the reservoir (= computational power)
## - instances_per_trial: number of reservoir trained instances to average the results/loss over
HYPER_SEARCH_CONFIG = {
    "exp": "brainhack",
    "hp_max_evals": 150, # -> use 200+ for more broader parameter exploration
    "hp_method": "random", # -> use "tpe" for bayesian optimization but without parellelization
    "seed": 42,
    "instances_per_trial": 3, # -> use 5 for more robustness
    "hp_space": {
        "N": ["choice", 500], # -> use 1000 or 2000 for more computational power
        "sr": ["loguniform", 1e-3, 3],
        "lr": ["loguniform", 1e-4, 1],
        "input_scaling": ["loguniform", 1e-3, 3], #"input_scaling": ["choice", 1.0],
        "ridge": ["loguniform", 1e-9, 1e3],
        "seed": ["choice", 1234],
        "warmup": ["choice", WARMUP],
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
HYPEROPT_SEARCH_DIR = Path("hyperopt-search")
# Plot parameters
HYPEROPT_PLOT_FIGSIZE = (24, 16) # (horizontal axis, vertical axis)
HYPEROPT_PLOT_DPI = 100
HYPEROPT_REPORT_PREFIX = "hyper-search-report"


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

    Y_pred = predict_test_series(
        esn,
        dataset["X_test_scaled"],
        dataset["scaler"],
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


def predict_test_series(esn, X_test_scaled, scaler):
    Y_pred_scaled = []

    for x in X_test_scaled:
        esn.reset()
        Y_pred_scaled.append(esn.run(x))

    return [
        scaler.inverse_transform(y_hat)
        for y_hat in Y_pred_scaled
    ]


def compute_metrics(Y_true_flat, Y_pred, Y_baseline_flat=None):
    Y_pred_flat = np.vstack(Y_pred)

    mse_reservoir = mean_squared_error(Y_true_flat, Y_pred_flat)
    rmse_reservoir = np.sqrt(mse_reservoir)
    mae_reservoir = mean_absolute_error(Y_true_flat, Y_pred_flat)
    r2_reservoir = r2_score(Y_true_flat, Y_pred_flat)

    metrics = {
        "mse": mse_reservoir,
        "rmse": rmse_reservoir,
        "mae": mae_reservoir,
        "r2": r2_reservoir,
    }

    if Y_baseline_flat is not None:
        mse_baseline = mean_squared_error(Y_true_flat, Y_baseline_flat)
        rmse_baseline = np.sqrt(mse_baseline)
        mae_baseline = mean_absolute_error(Y_true_flat, Y_baseline_flat)

        metrics.update(
            {
                "mse_baseline": mse_baseline,
                "rmse_baseline": rmse_baseline,
                "mae_baseline": mae_baseline,
                "rmse_ratio": rmse_reservoir / rmse_baseline,
            }
        )

    return metrics





# Define the objective function.
# ReservoirPy hyperopt objectives must take dataset and config first, then
# searched parameters as keyword-only arguments, and return a dict with "loss".
def objective(dataset, config, *, input_scaling, N, sr, lr, ridge, seed, warmup):
    instances = config["instances_per_trial"]
    variable_seed = int(seed)

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

        metrics = train_and_evaluate(params, dataset, warmup)
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


    movie_name = "AfterTheRain"
    target_column = "Throat"
    data_root = Path("../data/emofilm")
    behav_data = pd.read_csv(data_root / f"Annot_{movie_name}_stim.csv")

    timeseries_archive_path = data_root / f"{movie_name}_parcellated_timeseries.npz"

    X = np.load(data_root / f"{movie_name}_schaefer200_parcellated_timeseries.npz")
    X = [X[f'arr_{i}'] for i in range(len(X.files))]

    event_timing_dir = data_root / "event_timings"
    event_timing_files = sorted(event_timing_dir.glob(f"sub-*_{movie_name}_events.tsv"))
    if len(X) != len(event_timing_files):
        raise ValueError(
            f"Timeseries count ({len(X)}) does not match event timing files ({len(event_timing_files)})."
        )
    timing_file = event_timing_files[0]

    lag_seconds = 5.0

    movie_behavior = (
        behav_data[target_column]
        .astype(float)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy()
    )

    movie_target_length = len(movie_behavior)
    y_true_single = movie_behavior[1:]
    y_baseline_single = movie_behavior[:-1]

    x_sliced = [
        slice_fmri_to_movie_window(series, timing_file, tr=1.9)
        for series, timing_file in zip(X, event_timing_files)
    ]
    y_aligned = [
        align_behavior_to_fmri_trs(movie_behavior, timing_file, series.shape[0], lag_seconds=lag_seconds)
        for series, timing_file in zip(x_sliced, event_timing_files)
    ]
    # x_aligned = [
    #     align_fmri_to_movie_window(series, timing_file, movie_target_length, lag_seconds=lag_seconds)
    #     for series, timing_file in zip(X, event_timing_files)
    # ]
    X = x_sliced
    Y = y_aligned
    


    
    print(len(X)) #23
    print(Y[0].shape)
    print(X[0].shape) #(534, 200)
    print(X[1].shape) #(555, 200)
    print(y_aligned[1].shape) #(537, 200)
    # print(behav_data.shape) #(496, 50)




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

    # TODO: update starting here

    # Create one-step-ahead pairs: x[t] -> x[t+1].
    ## X training_data

    # sample one subject for now
    X = [X[0]]
    Y = [Y[0]]

    # identify TR corresponding to 70% of the movie duration
    split_index = int(0.7 * len(X))

    X_train = [s[:split_index] for s in X]
    Y_train = [s[:split_index] for s in Y]

    X_test = [s[split_index:] for s in X]
    Y_test = [s[split_index:] for s in Y]

    print(f"Training data: {len(X_train)} series, each with shape {X_train[0].shape}")
    print(f"Testing data: {len(X_test)} series, each with shape {X_test[0].shape}")
    print(f"Training target: {len(Y_train)} series, each with shape {Y_train[0].shape}")
    print(f"Testing target: {len(Y_test)} series, each with shape {Y_test[0].shape}")

    # Apply feature-wise MinMax normalization between -1 and 1.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.vstack(train_series))

    X_train_scaled = [scaler.transform(x) for x in X_train]
    Y_train_scaled = [scaler.transform(y) for y in Y_train]
    X_test_scaled = [scaler.transform(x) for x in X_test]

    # Naive baseline: predict x[t+1] = x[t].
    Y_baseline = X_test

    #------------------------------------------------------------------
    #------- END ZONE TO EDIT -----------------------------------------
    #------------------------------------------------------------------

    # Concatenate all test series to compute global metrics.
    Y_true_flat = np.vstack(Y_test)
    Y_baseline_flat = np.vstack(Y_baseline)

    # Configure research.
    hyperopt_config, experiment_dir = prepare_hyperopt_run(HYPER_SEARCH_CONFIG)

    # We precautionously save the configuration in a JSON file.
    config_path = experiment_dir / f"{hyperopt_config['exp']}.config.json"
    with open(config_path, "w+") as f:
        json.dump(hyperopt_config, f, indent=2)

    dataset = {
        "X_train_scaled": X_train_scaled,
        "Y_train_scaled": Y_train_scaled,
        "X_test_scaled": X_test_scaled,
        "Y_true_flat": Y_true_flat,
        "scaler": scaler,
        "workers": RESERVOIR_WORKERS,
    }

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
        dataset,
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

    alphas = np.logspace(-9, 3, 13)
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
    final_predictions = predict_test_series(final_esn, X_test_scaled, scaler)
    final_metrics = compute_metrics(Y_true_flat, final_predictions, Y_baseline_flat)

    # Print results.
    print("Reservoir + RidgeCV after ReservoirPy hyperopt")
    # print("-------------------")
    # print(f"MSE  : {final_metrics['mse']:.6f}")
    print(f"RMSE : {final_metrics['rmse']:.6f}")
    # print(f"MAE  : {final_metrics['mae']:.6f}")

    print()

    print("Persistence baseline: y[t+1] = x[t]")
    # print("------------------------------------")
    # print(f"MSE  : {final_metrics['mse_baseline']:.6f}")
    print(f"RMSE : {final_metrics['rmse_baseline']:.6f}")
    # print(f"MAE  : {final_metrics['mae_baseline']:.6f}")

    print()

    print("Ratio reservoir / baseline")
    # print("--------------------------")
    # print(f"MSE ratio  : {final_metrics['mse'] / final_metrics['mse_baseline']:.6f}")
    print(f"RMSE ratio : {final_metrics['rmse_ratio']:.6f}")
    # print(f"MAE ratio  : {final_metrics['mae'] / final_metrics['mae_baseline']:.6f}")

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
                ("lr", "sr", "input_scaling", "ridge"),
                metric="r2",
                loss_metric="loss",
                loss_behaviour="min",
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
