import yfinance as yf
import pandas as pd
import mplfinance as mpf

#from datar.all import f, select 
#from plotnine import ggplot, aes, geom_line, labs, theme_minimal


# Lista titoli Borsa Milano
tck = 'UCG.MI'
 


# Download dati
df = yf.download(
    "UCG.MI",
    start="2020-01-01",
    end="2024-03-01",
    auto_adjust=False
)

# flatten MultiIndex columns
df.columns = df.columns.get_level_values(0)
df.columns.name = None

# mplfinance richiede DateTimeIndex
df.index = pd.to_datetime(df.index)

# Candlestick chart
mpf.plot(
    df,
    type="candle",      # oppure "ohlc"
    style="yahoo",
    volume=True,
    title="UCG.MI",
    mav=(10, 20),
    figsize=(12, 7)
)


# Plot Close only
mpf.plot(
    df,
    type="line",
    style="yahoo",
    title="UCG.MI Close Price",
    figsize=(12, 6)
)



