import pandas as pd
import numpy as np
from plotnine import (ggplot,aes,geom_line,geom_point,labs,geom_abline,theme_minimal)
from datar.all import f, select, filter_, rename, group_by, arrange, ungroup, mutate, lag, slice_tail, tibble

from datar import options
options(backends=["pandas"])

# read cv results
cv = pd.read_csv("cv.csv")
cv["ds"] = pd.to_datetime(cv['ds'])



cv = (
    cv >> 
    mutate (id = f.unique_id, yhat = f.AutoNHITS) >>
    select(f.id, f.ds, f.y, f.yhat)
)


# Plot cv results
pl = (
    ggplot(cv) 
        + geom_point(aes(x="y", y="yhat"), color="blue")
        + geom_abline(intercept=0, slope=1, color="red", linetype="dashed")
        + theme_minimal()

)

pl.show()

pl = (
    ggplot(cv) 
        + geom_line(aes(x="ds", y="y") , color="black")
        + geom_point(aes(x="ds", y="y") , color="black")
        + geom_line(aes(x="ds", y="yhat") , color="blue")
        + geom_point(aes(x="ds", y="yhat") , color="blue")
        + theme_minimal()

)

pl.show()



# MAPE
from neuralforecast.losses.numpy import mape
mape = mape(y=cv.y.values,y_hat=cv.yhat.values)
print(mape)


cv
# Create trading signals based on the forecast
cv["y_1"] = (
    cv.groupby("id")["y"]
    .shift(1)
)


# Simple strategy: 
# go long if forecast is 10% above previous close
# short if 10% below
cv["position"] = np.where(
    cv["yhat"] >= 1.1* cv["y_1"],
    1,   # long
    -1   # short
)

# Calculate returns
cv["ret"] = (
    cv.groupby("id")["y"]
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

