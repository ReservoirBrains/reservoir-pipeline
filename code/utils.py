import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def compute_metrics(helpers, y_true_flat, y_pred, y_baseline_flat=None):
    metrics = dict(helpers.compute_metrics(y_true_flat, y_pred, y_baseline_flat))

    y_pred_flat = np.vstack(y_pred)
    metrics["corr"] = float(np.corrcoef(y_true_flat.ravel(), y_pred_flat.ravel())[0, 1])

    if y_baseline_flat is not None:
        metrics["corr_baseline"] = float(
            np.corrcoef(y_true_flat.ravel(), y_baseline_flat.ravel())[0, 1]
        )

        if "srmse" not in metrics:
            metrics["srmse"] = metrics["rmse"] / np.var(y_true_flat)
        if "srmse_baseline" not in metrics:
            metrics["srmse_baseline"] = metrics["rmse_baseline"] / np.var(y_true_flat)

    return metrics


def load_movie_timing(event_timing_path):
    event_table = pd.read_csv(event_timing_path)
    movie_rows = event_table.loc[event_table["trial_type"].eq("film")]
    if movie_rows.empty:
        raise ValueError(f"No film row found in {event_timing_path}")
    movie_row = movie_rows.iloc[0]
    movie_onset = float(movie_row["onset"])
    movie_duration = float(movie_row["duration"])
    run_end = float((event_table["onset"] + event_table["duration"]).max())
    return movie_onset, movie_duration, run_end


def slice_fmri_to_movie_window(x_series, event_timing_path, tr):
    movie_onset, movie_duration, _ = load_movie_timing(event_timing_path)
    start_index = int(round(movie_onset / tr))
    slice_length = int(round(movie_duration / tr))
    end_index = start_index + slice_length
    return x_series[start_index:end_index]


def align_fmri_to_movie_window(x_series, event_timing_path, target_length, lag_seconds=5.0):
    movie_onset, movie_duration, run_end = load_movie_timing(event_timing_path)
    x_time = np.linspace(0.0, run_end, num=len(x_series), endpoint=True)
    aligned_times = np.linspace(
        movie_onset + lag_seconds,
        movie_onset + movie_duration + lag_seconds,
        num=target_length,
        endpoint=False,
    )

    aligned = np.empty((target_length, x_series.shape[1]), dtype=float)
    for feature_index in range(x_series.shape[1]):
        aligned[:, feature_index] = np.interp(aligned_times, x_time, x_series[:, feature_index])
    return aligned


def align_behavior_to_fmri_trs(y_series, event_timing_path, target_length, lag_seconds=5.0):
    movie_onset, movie_duration, _ = load_movie_timing(event_timing_path)

    y_series = np.asarray(y_series)
    was_1d = y_series.ndim == 1
    if was_1d:
        y_series = y_series[:, None]

    behavior_time = np.linspace(
        movie_onset,
        movie_onset + movie_duration,
        num=len(y_series),
        endpoint=False,
    )

    fmri_times = np.linspace(
        movie_onset + lag_seconds,
        movie_onset + movie_duration + lag_seconds,
        num=target_length,
        endpoint=False,
    )

    aligned = np.empty((target_length, y_series.shape[1]), dtype=float)
    for feature_index in range(y_series.shape[1]):
        aligned[:, feature_index] = np.interp(fmri_times, behavior_time, y_series[:, feature_index])

    if was_1d:
        return aligned[:, 0]
    return aligned


def train_and_evaluate_fold(params, dataset, train_positions, val_positions, helpers, build_esn, predict_test_series):
    esn = build_esn(**params)
    x_train_fold = [dataset["X_train_scaled_all"][i] for i in train_positions]
    y_train_fold = [dataset["Y_train_scaled_all"][i] for i in train_positions]
    x_val_fold = [dataset["X_train_scaled_all"][i] for i in val_positions]
    y_true_fold = np.hstack([dataset["y_true_single"] for _ in val_positions])

    esn.fit(
        x_train_fold,
        y_train_fold,
        warmup=dataset["warmup"],
        workers=dataset["workers"],
    )

    y_pred = predict_test_series(esn, x_val_fold, dataset["y_scaler"])
    return compute_metrics(helpers, y_true_fold, y_pred)


def objective(dataset, config, *, input_scaling, N, sr, lr, ridge, seed, helpers, build_esn, predict_test_series):
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

        fold_metrics = [
            train_and_evaluate_fold(params, dataset, train_positions, val_positions, helpers, build_esn, predict_test_series)
            for train_positions, val_positions in dataset["inner_splits"]
        ]

        rmses.append(float(np.mean([metrics["rmse"] for metrics in fold_metrics])))
        maes.append(float(np.mean([metrics["mae"] for metrics in fold_metrics])))
        r2s.append(float(np.mean([metrics["r2"] for metrics in fold_metrics])))
        variable_seed += 1

    return {
        "loss": float(np.mean(rmses)),
        "rmse": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae": float(np.mean(maes)),
        "r2": float(np.mean(r2s)),
    }