
import yfinance as yf
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE

from ray import tune


#from lightgbm import LGBMRegressor
#from mlforecast import MLForecast
#from mlforecast.target_transforms import Differences

#from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    labs,
    geom_abline,
    theme_minimal
)

# =========================================================
# Functions
# =========================================================
def split(df, h):
    
    df_trn = df.iloc[:-h].copy()
    df_tst = df.iloc[-h:].copy()

    return df_trn, df_tst

# =========================================================
# PARAMETERS
# =========================================================

tck = "UCG.MI"
initial_capital = 100.0

start_date = "2020-01-01"
end_date = "2026-04-30"


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

ts = df.copy()
ts["unique_id"] = tck
ts["ds"] = ts['Date']
ts["y"] = ts["Close"]
ts["Volume"] = ts["Volume"].shift(1)  # Use previous day's volume as a feature

ts = ts[["unique_id", "ds", "y"]]

# =========================================================
# Split train/test
# =========================================================

"""
# =========================================================
# MODEL
# =========================================================


season_length = 1  

model = [AutoARIMA(season_length=season_length)]

sf = StatsForecast(
    models = model,
    freq = 'B',
    n_jobs = 1
)

# =========================================================
# CROSS VALIDATION
# 1-day ahead forecast 
# =========================================================

cv = sf.cross_validation(
    df=ts,
    h=1,
    n_windows=120,
    refit=True
)
"""


# =========================================================
# MODEL
# =========================================================

h = 1

config_nhits = {
    "input_size": tune.choice([20, 40, 60]),
    "max_steps": tune.choice([300, 500]),
    "learning_rate": tune.choice([1e-3, 5e-4]),
    "batch_size": tune.choice([16, 32]),
    "windows_batch_size": tune.choice([128, 256]),

    # meno aggressivi del default
    "n_pool_kernel_size": tune.choice([[2, 2, 1], [4, 2, 1]]),
    "n_freq_downsample": tune.choice([[4, 2, 1], [8, 4, 1]]),

    "random_seed": tune.choice([42]),
}

models = [
    AutoNHITS(
        h=h,
        loss=MAE(),
        valid_loss=MAE(),
        config=config_nhits,
        num_samples=5,
        cpus=1,
        gpus=0,
        verbose=True,
    )
]

nf = NeuralForecast(
    models=models,
    freq="B"
)

cv = nf.cross_validation(
    df=ts,
    h=h,
    n_windows=120,
    step_size=1,
    refit=True
)



# Plot cv results
pl = (
    ggplot(cv) 
        + geom_point(aes(x="y", y="AutoARIMA"), color="blue")
        + geom_abline(intercept=0, slope=1, color="red", linetype="dashed")
        + theme_minimal()

)

pl.show()




# Create trading signals based on the forecast
cv["y_1"] = (
    cv.groupby("unique_id")["y"]
    .shift(1)
)


# Simple strategy: 
# go long if forecast is 10% above previous close
# short if 10% below
cv["position"] = np.where(
    cv["AutoARIMA"] >= 1.1* cv["y_1"],
    1,   # long
    -1   # short
)

# Calculate returns
cv["ret"] = (
    cv.groupby("unique_id")["y"]
    .pct_change()
)

cv["strategy_ret"] = (
    cv["position"].shift(1) * cv["ret"]
)

# equity curve
cv["equity"] = (
    1 + cv["strategy_ret"]
).cumprod()


pl = (
    ggplot(cv) 
        + geom_line(aes(x="ds", y="equity"), color="blue")
        + theme_minimal()

)
pl.show()


