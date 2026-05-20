import os
import warnings

import torch

import yfinance as yf
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE
from ray import tune


from plotnine import (ggplot,aes,geom_line,geom_point,labs,geom_abline,theme_minimal)
import time

# =========================================================
# SETTINGS
# =========================================================
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =========================================================
# CUDA CHECK
# =========================================================
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

# =========================================================
# PARAMETERS
# =========================================================
tck = "UCG.MI"
initial_capital = 100.0

start_date = "2020-01-01"
end_date = "2026-04-30"

h = 1

# =========================================================
# DOWNLOAD DATA
# =========================================================
df = (
    yf.download(
        tickers=tck,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True
    )
    .reset_index()
)

# Flatten yfinance columns
df.columns = df.columns.get_level_values(0)
df.columns.name = None

df = (
    df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    .dropna()
    .drop_duplicates(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)

# =========================================================
# PREPARE DATA FOR NIXTLA / MLFORECAST
# =========================================================
# Correctly include Volume column in ts DataFrame
ts = df.copy()
ts["unique_id"] = tck
ts["ds"] = ts['Date']
ts["y"] = ts["Close"]
ts["Volume"] = ts["Volume"].shift(1)  # Use previous day's volume as a feature
ts["Open"] = ts["Open"].shift(1)  # Use previous day's volume as a feature
ts["High"] = ts["High"].shift(1)  # Use previous day's volume as a feature
ts["Low"] = ts["Low"].shift(1)  # Use previous day's volume as a feature
ts = ts[["unique_id", "ds", "y", "Volume", "High", "Low", "Open"]]  # Ensure Volume is included

# Drop rows with missing values after shifting
ts = ts.dropna(subset=["Volume"])
ts = ts.dropna(subset=["High"])
ts = ts.dropna(subset=["Low"])
ts = ts.dropna(subset=["Open"])

# Include Volume as an exogenous variable
exogenous_features = ["Volume", "High", "Low", "Open"]  # List of exogenous features
ts = ts[["unique_id", "ds", "y"] + exogenous_features]  # Add exogenous features to the DataFrame

# =========================================================
# MODEL CONFIG
# =========================================================
config_nhits = {
    "input_size": tune.choice([20, 40, 60, 80, 100, 120, 140, 160, 180, 200]),
    "max_steps": tune.choice([300, 500, 700]),
    "learning_rate": tune.choice([1e-3, 5e-4, 1e-4]),
    "batch_size": tune.choice([16, 32, 64]),
    "windows_batch_size": tune.choice([128, 256, 512]),

    "n_pool_kernel_size": tune.choice([[2, 2, 1], [3, 2, 1]]),
    "n_freq_downsample": tune.choice([[8, 4, 1], [4, 2, 1]]),

    "scaler_type": tune.choice(["robust", "standard"]),
    "random_seed": tune.choice([42, 123, 2026]),
}

# =========================================================
# MODEL INSTANTIATION
# =========================================================
models = [
    AutoNHITS(
        h=h,
        loss=MAE(),
        valid_loss=MAE(),
        config=config_nhits,
        num_samples=5,
        cpus=1,
        gpus=1,
        verbose=False,
        backend="ray"
    )
]

# Adjust NeuralForecast instantiation to remove unsupported arguments
nf = NeuralForecast(
    models=models,
    freq="B"  # Frequency remains specified
)

# =========================================================
# CROSS VALIDATION
# =========================================================
start = time.time()
cv = nf.cross_validation(
    df=ts,
    h=h,
    n_windows=12,
    step_size=1,
    refit=True
)
elapsed = (time.time() - start)/60
print(f"Elapsed time: {elapsed:.2f} minutes")
cv.to_csv("cv.csv", index=False)

