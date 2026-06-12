import numpy as np

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, ScikitLearnNode

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.base",
)

# Reproducibility.
np.random.seed(42)
rpy.set_seed(42)


# Create 10 random time series with shape (4000, 200).
print("Predicting random data for demo purpose for 10 individuals with 200 voxels on 4,000 time points ...\n")
# NB: Timeseries of differents subjects don't have to be of same length
series_list = [
    np.random.randn(4000, 200).astype(np.float32)
    for _ in range(10)
]

# Split train/test by full time series.
print("Splitting subjects 8 for training and 2 for testing")
train_series = series_list[:8]
test_series = series_list[8:]


# Create one-step-ahead pairs: x[t] -> x[t+1].
X_train = [s[:-1] for s in train_series]
Y_train = [s[1:] for s in train_series]

X_test = [s[:-1] for s in test_series]
Y_test = [s[1:] for s in test_series]


# Apply feature-wise MinMax normalization between -1 and 1.
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(np.vstack(train_series))

X_train_scaled = [scaler.transform(x) for x in X_train]
Y_train_scaled = [scaler.transform(y) for y in Y_train]
X_test_scaled = [scaler.transform(x) for x in X_test]


# Create the reservoir.
reservoir = Reservoir(
    units=300,
    sr=0.9,
    lr=0.3,
    input_scaling=0.5,
)


# Create the scikit-learn readout with RidgeCV, following the ReservoirPy tutorial pattern.
alphas = np.logspace(-9, 3, 13)
readout = ScikitLearnNode(
    model=RidgeCV,
    alphas=alphas,
)
print("Regularization: Alphas tested:", np.array2string(alphas, formatter={"float_kind": lambda x: f"{x:.1e}"}))
print()

# Build the ESN: reservoir followed by readout.
esn = reservoir >> readout


# Train the model.
esn.fit(
    X_train_scaled,
    Y_train_scaled,
    warmup=20,
)


# Predict on the test time series.
Y_pred_scaled = []

for x in X_test_scaled:
    esn.reset()
    Y_pred_scaled.append(esn.run(x))


# Transform predictions back to the original scale.
Y_pred = [
    scaler.inverse_transform(y_hat)
    for y_hat in Y_pred_scaled
]


# Naive baseline: predict x[t+1] = x[t].
Y_baseline = X_test


# Concatenate all test series to compute global metrics.
Y_true_flat = np.vstack(Y_test)
Y_pred_flat = np.vstack(Y_pred)
Y_baseline_flat = np.vstack(Y_baseline)


# Compute reservoir metrics.
mse_reservoir = mean_squared_error(Y_true_flat, Y_pred_flat)
rmse_reservoir = np.sqrt(mse_reservoir)
mae_reservoir = mean_absolute_error(Y_true_flat, Y_pred_flat)


# Compute baseline metrics.
mse_baseline = mean_squared_error(Y_true_flat, Y_baseline_flat)
rmse_baseline = np.sqrt(mse_baseline)
mae_baseline = mean_absolute_error(Y_true_flat, Y_baseline_flat)


# Print results.
print("Reservoir + RidgeCV")
#print("-------------------")
#print(f"MSE  : {mse_reservoir:.6f}")
print(f"RMSE : {rmse_reservoir:.6f}")
#print(f"MAE  : {mae_reservoir:.6f}")

print()

print("Persistence baseline: y[t+1] = x[t]")
#print("------------------------------------")
#print(f"MSE  : {mse_baseline:.6f}")
print(f"RMSE : {rmse_baseline:.6f}")
#print(f"MAE  : {mae_baseline:.6f}")

print()

print("Ratio reservoir / baseline")
#print("--------------------------")
#print(f"MSE ratio  : {mse_reservoir / mse_baseline:.6f}")
print(f"RMSE ratio : {rmse_reservoir / rmse_baseline:.6f}")
#print(f"MAE ratio  : {mae_reservoir / mae_baseline:.6f}")


# Print the alpha selected by RidgeCV.
print()
print(f"Regularization: Alpha selected by RidgeCV: {readout.instances.alpha_:.1e}")