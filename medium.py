# source: https://medium.com/@marcelboersma/neuralforecasting-for-finance-predict-stock-values-d880db4ca4a1

import yfinance as yf
import pandas as pd


tck = ["AAPL", "GOOG", "MSFT"]
#initial_capital = 100.0

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

# Only keep the 'Close' price for each ticker
df = df['Close'] 


# Convert the Series to a DataFrame if it's not already (optional but recommended)
if isinstance(df, pd.Series):
    df = df.to_frame()


# Melt the dataframe to long format
hist = df.melt(ignore_index=False, var_name='Ticker', value_name='Close')
hist.reset_index(inplace=True)


hist.rename(columns={'Date': 'ds', 'Ticker': 'unique_id', 'Close':'y'}, inplace=True)
#hist.head()






# Flatten yfinance columns
df.columns = df.columns.get_level_values(0)
df.columns.name = None



# Fetch data for multiple tickers
#tickers = ["AAPL", "GOOG", "MSFT"]
#data = yf.download(tickers, start="2015-01-01", end="2023-06-30")

# Reshape the data
df = data['Adj Close']  # No need to unstack here

# Convert the Series to a DataFrame if it's not already (optional but recommended)
if isinstance(df, pd.Series):
    df = df.to_frame()

# Melt the dataframe to long format
hist = df.melt(ignore_index=False, var_name='Ticker', value_name='Adj Close')
hist.reset_index(inplace=True)

hist.rename(columns={'Date': 'ds', 'Ticker': 'unique_id', 'Adj Close':'y'}, inplace=True)
hist.head()