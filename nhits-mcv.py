import os
import warnings

import torch

import yfinance as yf
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE, MAPE
from ray import tune


#from plotnine import (ggplot,aes,geom_line,geom_point,labs,geom_abline,theme_minimal)
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

test = True
tck_list = ["UCG.MI", "ISP.MI"]
start_date = "2020-01-01"
end_date = "2026-04-30"
h = 1



# =========================================================
# DOWNLOAD DATA FOR MULTIPLE TICKERS
# =========================================================
dfs = []
for tck in tck_list:
    df_tck = (
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
    df_tck.columns = df_tck.columns.get_level_values(0)
    df_tck.columns.name = None
    df_tck = (
        df_tck[["Date", "Open", "High", "Low", "Close", "Volume"]]
        .dropna()
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df_tck["unique_id"] = tck
    dfs.append(df_tck)

# Concatenate all tickers
df = pd.concat(dfs, ignore_index=True)


# =========================================================
# PREPARE DATA FOR NIXTLA / MLFORECAST
# =========================================================
ts = df.copy()
ts["ds"] = ts['Date']
ts["y"] = ts["Close"]
ts["Volume"] = ts.groupby("unique_id")['Volume'].shift(1)  # Use previous day's volume as a feature per ticker
ts = ts[["unique_id", "ds", "y"]]

# =========================================================
# MODEL CONFIG
# =========================================================
config_nhits = {
    "input_size": tune.choice([20, 40, 60, 80]),
    "max_steps": tune.choice([300, 500, 700]),
    "learning_rate": tune.choice([1e-3, 5e-4, 1e-4]),
    "batch_size": tune.choice([16, 32, 64, 128, 256, 512]),
    "windows_batch_size": tune.choice([32, 64, 128, 256, 512]),

    "n_pool_kernel_size": tune.choice([[2, 2, 1], [3, 2, 1]]),
    "n_freq_downsample": tune.choice([[8, 4, 1], [4, 2, 1]]),

    "scaler_type": tune.choice(["robust", "standard"]),
    "random_seed": tune.choice([42, 123, 2026]),
}

# Quick test config for fast runs
config_test = {
    'input_size': 1,
    'max_steps': 1,
    'learning_rate': 0.01,
    'batch_size': 1,
    'windows_batch_size': 1,
    'n_pool_kernel_size': [1, 1, 1],
    'n_freq_downsample': [1, 1, 1],
    'scaler_type': 'standard',
    'random_seed': 42
}


# Use config_test for quick testing if test is True
if test:
    config_nhits = config_test

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

nf = NeuralForecast(
    models=models,
    freq="B"
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

# Calculate MAPE using nixtla's implementation
if "AutoNHITS" in cv.columns:
    mape_value = MAPE()(np.array(cv["y"]), np.array(cv["AutoNHITS"]))
    print(f"MAPE: {mape_value:.2f}%")

