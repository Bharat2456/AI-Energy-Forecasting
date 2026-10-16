"""Feature engineering: turn a time series into a supervised-learning table.

Key rule: **no data leakage**. Every lag / rolling feature only looks at data that is
at least FORECAST_MIN_LAG (=24) hours old, i.e. what we would really know when making a
day-ahead forecast at midnight.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import LAGS, ROLL_WINDOWS, FORECAST_MIN_LAG, TARGET


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df has a DatetimeIndex and columns energy_mwh, temperature_c, humidity_pct, is_holiday."""
    out = pd.DataFrame(index=df.index)
    idx = df.index

    # --- calendar features
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    out["day_of_year"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = df["is_holiday"].astype(int)
    # cyclical encoding: hour 23 is close to hour 0, December close to January
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    # --- weather features (known / forecastable ahead of time)
    out["temperature_c"] = df["temperature_c"]
    out["humidity_pct"] = df["humidity_pct"]
    out["cooling_degrees"] = (df["temperature_c"] - 24).clip(lower=0)
    out["heating_degrees"] = (16 - df["temperature_c"]).clip(lower=0)

    # --- history features (lags and rolling statistics of past demand)
    for lag in LAGS:
        out[f"lag_{lag}h"] = df[TARGET].shift(lag)
    past = df[TARGET].shift(FORECAST_MIN_LAG)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}h"] = past.rolling(w).mean()
    out["roll_std_24h"] = past.rolling(24).std()
    return out


FEATURE_COLUMNS = list(make_features(pd.DataFrame(
    {TARGET: [0.0] * 3, "temperature_c": [0.0] * 3, "humidity_pct": [0.0] * 3, "is_holiday": [0] * 3},
    index=pd.date_range("2000-01-01", periods=3, freq="h"))).columns)


def build_dataset(df: pd.DataFrame):
    """Return (X, y) with rows that have complete history (first ~8 days are dropped)."""
    feats = make_features(df)
    mask = feats.notna().all(axis=1)
    return feats.loc[mask, FEATURE_COLUMNS], df.loc[mask, TARGET]
