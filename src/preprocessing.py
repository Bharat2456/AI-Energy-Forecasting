"""Data cleaning: make the raw CSV safe for machine learning.

Steps (each one is reported in the returned `report` dict):
1. parse timestamps, drop duplicates, sort
2. rebuild a *complete* hourly index (so missing hours become visible NaNs)
3. detect outliers/sensor glitches with a rolling-median rule -> mark as NaN
4. fill gaps by time interpolation
5. recompute the holiday flag from the calendar
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import TARGET, TIME_COL, WEATHER_COLS
from .data_generator import is_holiday

REQUIRED_COLUMNS = [TIME_COL, TARGET] + WEATHER_COLS


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}. "
                         f"Expected columns: {REQUIRED_COLUMNS}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def clean_data(df: pd.DataFrame, outlier_low: float = 0.5, outlier_high: float = 1.6):
    """Return (clean_df indexed by timestamp, report dict)."""
    report = {"rows_in": int(len(df))}
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    report["duplicates_removed"] = int(df.duplicated(subset=TIME_COL).sum())
    df = df.drop_duplicates(subset=TIME_COL).sort_values(TIME_COL).set_index(TIME_COL)

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name=TIME_COL)
    report["missing_timestamps_added"] = int(len(full_index) - len(df))
    df = df.reindex(full_index)

    # --- outliers: compare each hour with the rolling median of the surrounding 25 hours
    median = df[TARGET].rolling(25, center=True, min_periods=5).median()
    ratio = df[TARGET] / median
    outliers = (ratio > outlier_high) | (ratio < outlier_low)
    report["outliers_flagged"] = int(outliers.sum())
    df.loc[outliers, TARGET] = np.nan

    report["missing_values_before"] = {c: int(df[c].isna().sum()) for c in [TARGET] + WEATHER_COLS}
    for c in [TARGET] + WEATHER_COLS:
        df[c] = df[c].interpolate(method="time", limit_direction="both")
    df["humidity_pct"] = df["humidity_pct"].clip(0, 100)
    df[TARGET] = df[TARGET].clip(lower=0)
    report["missing_values_after"] = {c: int(df[c].isna().sum()) for c in [TARGET] + WEATHER_COLS}

    df["is_holiday"] = is_holiday(df.index)
    report["rows_out"] = int(len(df))
    report["start"] = str(df.index.min())
    report["end"] = str(df.index.max())
    return df[[TARGET] + WEATHER_COLS + ["is_holiday"]], report
