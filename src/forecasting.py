"""Future forecasting with a *recursive* strategy.

To predict hour t we need demand at t-24, t-48 ... (lag features). For horizons > 24 h
those lags are themselves forecasts, so we predict one hour at a time and feed each
prediction back into the history. Future weather = a seasonal "climatology" (average
temperature for that week-of-year and hour) - in a real system you would plug in a
weather-service forecast instead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import FORECAST_HORIZON_HOURS, TARGET, WEATHER_COLS
from .data_generator import is_holiday
from .features import FEATURE_COLUMNS, make_features

_TAIL = 400  # rows of history needed (>= 168 + 24 for the longest rolling window)


def climatology_weather(history: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Typical weather for each future hour = mean over history for same (week-of-year, hour)."""
    key = [history.index.dayofyear // 7, history.index.hour]
    table = history[WEATHER_COLS].groupby(key).mean()
    fkey = pd.MultiIndex.from_arrays([future_index.dayofyear // 7, future_index.hour])
    out = table.reindex(fkey)
    out = out.fillna(history[WEATHER_COLS].mean())
    out.index = future_index
    return out


def recursive_forecast(model, history: pd.DataFrame, horizon: int = FORECAST_HORIZON_HOURS,
                       future_weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Forecast `horizon` hours after the last timestamp of `history`.

    Returns DataFrame[forecast_mwh, temperature_c, humidity_pct] indexed by timestamp.
    """
    future_index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    if future_weather is None:
        future_weather = climatology_weather(history, future_index)
    future_weather = future_weather.reindex(future_index)

    buf = history[[TARGET] + WEATHER_COLS + ["is_holiday"]].tail(_TAIL).copy()
    preds = []
    for ts in future_index:
        row = {TARGET: np.nan, **future_weather.loc[ts, WEATHER_COLS].to_dict(),
               "is_holiday": int(is_holiday(pd.DatetimeIndex([ts]))[0])}
        buf.loc[ts] = row
        x = make_features(buf.tail(_TAIL)).iloc[[-1]][FEATURE_COLUMNS]
        yhat = max(float(model.predict(x)[0]), 0.0)
        buf.loc[ts, TARGET] = yhat
        preds.append(yhat)
    out = future_weather[WEATHER_COLS].copy()
    out.insert(0, "forecast_mwh", np.round(preds, 2))
    out.index.name = "timestamp"
    return out


def recursive_backtest(model, clean: pd.DataFrame, test_start, test_end, horizon=FORECAST_HORIZON_HOURS,
                       step_days=21, max_windows=4):
    """Honest multi-day check: pretend we are at several past dates and forecast 7 days ahead
    using only the data before that date (actual weather is used as a stand-in for a good
    weather forecast). Returns list of dicts with RMSE / MAPE per window."""
    from .models import regression_metrics
    rows, start = [], pd.Timestamp(test_start)
    while len(rows) < max_windows and start + pd.Timedelta(hours=horizon) <= pd.Timestamp(test_end):
        hist = clean.loc[: start - pd.Timedelta(hours=1)]
        idx = pd.date_range(start, periods=horizon, freq="h")
        fc = recursive_forecast(model, hist, horizon, clean.loc[idx, WEATHER_COLS])
        m = regression_metrics(clean.loc[idx, TARGET], fc["forecast_mwh"])
        rows.append({"start": str(start), **m})
        start += pd.Timedelta(days=step_days)
    return rows
