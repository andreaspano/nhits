# =========================================================
# INSTALL
# pip install yfinance pandas mlforecast lightgbm scikit-learn
# =========================================================

import yfinance as yf
import pandas as pd

from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from plotnine import * 

# =========================================================
# PARAMETERS
# =========================================================

tck = "UCG.MI"

start_date = "2022-01-01"
end_date = "2025-01-01"



# =========================================================
# DOWNLOAD DATA
# =========================================================

df = (
    yf.download(
        "UCG.MI",
        start=start_date,
        end=end_date,
        auto_adjust=True
    )
    .reset_index()
)

# flatten columns
df.columns = df.columns.get_level_values(0)
df.columns.name = None

# =========================================================
# PREPARE DATA
# =========================================================

ts = (
    df[["Date", "Close", "Volume"]]
    .dropna()
    .drop_duplicates(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)

# create lagged volume
ts["volume_lag1"] = ts["Volume"].shift(1)

# remove first NA row
ts = ts.dropna().reset_index(drop=True)

# keep original dates
date_map = ts[["Date"]].copy()
date_map["ds"] = range(len(date_map))

# Nixtla format
ts["unique_id"] = "UCG"
ts["ds"] = range(len(ts))
ts["y"] = ts["Close"]

# final dataset
ts = ts[[
    "unique_id",
    "ds",
    "y",
    "volume_lag1"
]]


# =========================================================
# MODEL
# =========================================================

model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

fcst = MLForecast(
    models=[model],

    freq=1,

    lags=[1, 2, 5, 10, 20],

    target_transforms=[
        Differences([1])
    ]
)


# =========================================================
# CROSS VALIDATION
# =========================================================

cv = fcst.cross_validation(
    df=ts,

    h=1,
    n_windows=60,

    static_features=[],

    refit=True
)

# merge dates back
cv = cv.merge(date_map, on="ds", how="left")

print(cv.head())

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(cv["y"], cv["LGBMRegressor"])
rmse = mean_squared_error(cv["y"], cv["LGBMRegressor"]) ** 0.5
mape = mean_absolute_percentage_error(cv["y"], cv["LGBMRegressor"]) 


print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")




# =========================================================
# OPTIONAL: SAVE RESULTS
# =========================================================

from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_abline,
    labs,
    theme_minimal
)

# scatter plot actual vs predicted
p = (
    ggplot(cv,aes(x="y",y="LGBMRegressor" ))
    + geom_point(alpha=0.7)
    # 45 degree line
    + geom_abline(slope=1,intercept=0,linetype="dashed")
    + labs(title="Actual vs Predicted",x="Actual",y="Predicted")
    + theme_minimal()
)

p.show()


##################


# cv contiene: unique_id, Date, ds, cutoff, y, LGBMRegressor
plot_df = (
    cv[["Date", "y", "LGBMRegressor"]]
    .rename(columns={
        "y": "Actual",
        "LGBMRegressor": "Forecast"
    })
    .melt(
        id_vars="Date",
        value_vars=["Actual", "Forecast"],
        var_name="series",
        value_name="value"
    )
)

p = (
    ggplot(plot_df, aes(x="Date", y="value", color="series"))
    + geom_line()
    + geom_point(size=1.8)
    + labs(
        title="Time Series: Actual vs Forecast",
        x="Date",
        y="Close price",
        color=""
    )
    + theme_minimal()
)

p.show()


# =========================================================
# STRATEGY
# Buy at Open if Forecast_Close > Open
# Sell at Close
# Otherwise stay cash
# =========================================================

initial_capital = 100.0

cv["signal"] = cv["Forecast_Close"] > cv["Open"]

cv["daily_return"] = 1.0
cv.loc[cv["signal"], "daily_return"] = (
    cv.loc[cv["signal"], "Actual_Close"] /
    cv.loc[cv["signal"], "Open"]
)

cv["capital"] = initial_capital * cv["daily_return"].cumprod()
cv["profit"] = cv["capital"] - initial_capital
cv["return_pct"] = cv["capital"] / initial_capital - 1


# =========================================================
# RESULTS
# =========================================================

final_capital = cv["capital"].iloc[-1]
profit = final_capital - initial_capital
return_pct = profit / initial_capital

print(f"Initial capital: €{initial_capital:.2f}")
print(f"Final capital:   €{final_capital:.2f}")
print(f"Profit:          €{profit:.2f}")
print(f"Return:          {return_pct:.2%}")

print(cv[[
    "Date",
    "Open",
    "Actual_Close",
    "Forecast_Close",
    "signal",
    "daily_return",
    "capital",
    "profit",
    "return_pct"
]].tail())


# =========================================================
# SAVE DAILY CAPITAL
# =========================================================

cv[[
    "Date",
    "Open",
    "Actual_Close",
    "Forecast_Close",
    "signal",
    "daily_return",
    "capital",
    "profit",
    "return_pct"
]].to_csv("daily_capital_ml_strategy.csv", index=False)