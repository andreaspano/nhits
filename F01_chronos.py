import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline
import yfinance as yf


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    labs,
    theme_minimal
)

def split(df, h):
    
    df_trn = df.iloc[:-h].copy()
    df_tst = df.iloc[-h:].copy()

    return df_trn, df_tst

# =========================================================
# PARAMETERS
# =========================================================

tck = "UCG.MI"
initial_capital = 100.0

start_date = "2022-01-01"
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
    df[["Date", "Close"]]
    .dropna()
    .drop_duplicates(subset=["Date"])
    .sort_values("Date")
    .reset_index(drop=True)
)


# Reindex to ensure no missing dates for the given frequency
full_index = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='B')
df = df.set_index('Date').reindex(full_index).rename_axis('Date').reset_index()

# Fill missing values (if any)
#df['unique_id'] = df_trn['unique_id'].fillna(method='ffill')  # Forward fill unique_id
df['Close'] = df['Close'].fillna(method='ffill')  # Forward fill unique_id

#df["y"] = np.log(df["Close"]).diff()
df["y"] = df["Close"]


# =========================================================
# PREPARE DATA FOR NIXTLA / MLFORECAST
# target = Close
# dynamic feature = Volume
# autoregressive features = lags of Close
# =========================================================

ts = df.copy()
ts["unique_id"] = "UCG"
ts["ds"] = ts["Date"]
ts["y"] = ts["y"]

ts = ts[["unique_id", "ds", "y"]]


h = 12 #Forecast horizon 


# Simple plot
(
    ggplot(ts) 
    + geom_line(aes(x = 'ds', y = 'y'), size = 1)
)


# Split trn & tst
df_trn, df_tst = split(ts, h)
v = df_tst['ds'].min()

#df_h = df_tst.drop(columns="y")


from neuralforecast.tsdataset import TimeSeriesDataset
dataset, *_ = TimeSeriesDataset.from_df(df_trn, time_col="ds", target_col="y", id_col="unique_id")


from neuralforecast.auto import AutoRNN

# Use your own config or AutoRNN.default_config
config = dict(max_steps=1, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoRNN(h=12, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset, val_size=12)
y_hat = model.predict(dataset=dataset)












pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

df_fct = pipeline.predict_df(
    df_trn,
    future_df=df_h,
    prediction_length=h,  # Number of steps to forecast
    #quantile_levels=[0.1, 0.5, 0.9],  # Quantile for probabilistic forecast
    id_column="unique_id",  # Column identifying different time series
    timestamp_column="ds",  # Column with datetime information
    target="y",  # Column(s) with time series values to predict
 
)


#pl = nixtla_client.plot(df, time_col='ds', target_col='y')


df_fct = nixtla_client.forecast(
    df=df_trn,
    h=h,
    freq='MS',
    time_col='ds',
    target_col='y'
)



df_merged = pd.merge(
        df_tst,
        df_fct,
        #on=["unique_id", "ds"],
        on="ds",
        how="left"
    )

# Exclude TOTAL series unless requested
df_plt = df_merged
    
# build plotnine object: actuals in default color, forecasts in red
(
    ggplot(df)  # dataset for actuals
        + aes(x="ds", y="y")
        + geom_line()
        + geom_line(aes(x="ds", y="y"), data=df_merged, color="blue")
        + geom_line(aes(x="ds", y="TimeGPT"), data=df_merged, color="red")
        + geom_vline(xintercept=v, linetype="dashed", color="black")
)
            
            
