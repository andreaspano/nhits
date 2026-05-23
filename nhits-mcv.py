import os
import warnings

import torch

import yfinance as yf
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE, MAPE
from neuralforecast.losses.numpy  import  mape

from ray import tune

from datetime import datetime
from pathlib import Path

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
test = False
tck_list = ["UCG.MI", "ISP.MI", "BAMI.MI", "BPE.MI", "BMPS.MI", "FBK.MI", "MB.MI", "G.MI", "AZM.MI", "UNI.MI"]
start_date = "2020-01-01"
end_date = "2026-04-30"
h = 1
# =========================================================
# OUTPUT SETTINGS
# =========================================================
out_dir = Path("./out")
out_dir.mkdir(parents=True, exist_ok=True)
tag = datetime.now().strftime('%Y_%m_%d_%H_%M')
file_cv = out_dir /  f"df_cv_{tag}.csv"
file_mape = out_dir /  f"df_mape_{tag}.csv"

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
        cpus=6,
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
df_cv = nf.cross_validation(
    df=ts,
    h=h,
    n_windows=12,
    step_size=1,
    refit=True
)
elapsed = (time.time() - start)/60




print(f"Elapsed time: {elapsed:.2f} minutes")

# Save cross-validation results to csv
df_cv.to_csv(file_cv, index=False)



# read cv results back from csv (optional, can be skipped if df_cv is still in memory)
#df_cv = pd.read_csv(file_cv)

# prepare cv results
df_cv = (
    df_cv
    .assign(
        ds=pd.to_datetime(df_cv.ds),
        id=df_cv.unique_id,
        yhat=df_cv.AutoNHITS
    )
    .filter(items=["ds", "id", "y", "yhat"])
)

# Global MAPE
global_mape = mape(df_cv["y"].values, df_cv["yhat"].values)

# MAPE by id
df_mape = (
    df_cv
    .groupby("id")[["y", "yhat"]]
    .apply(
        lambda x: mape(
            x["y"].values,
            x["yhat"].values
        )
    )
    .reset_index(name="MAPE")
)


df_mape.loc[len(df_mape)] = ["GLOBAL", global_mape]


# save mape results to csv
df_mape.to_csv(file_mape, index=False)
