"""Virtual simulation of a smart-city district's electricity demand.

We have no access to a real utility's SCADA / smart-meter data, so we *simulate* it
using the same drivers that shape real demand:

* daily double-hump profile (morning + evening peak),
* weekday / weekend / holiday behaviour,
* weather: hot afternoons -> air-conditioning load (cooling degree effect),
* slow yearly growth (urbanisation),
* random noise, plus realistic *data problems* (missing hours, sensor spikes,
  gaps in timestamps) so the preprocessing stage has real work to do.

The generator is seeded, so the same seed always gives the same CSV.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import (DEFAULT_DAYS, DEFAULT_SEED, HOLIDAYS, START_DATE, TARGET,
                     TIME_COL)


def is_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    """1 if the date is one of the fixed-date holidays in config.HOLIDAYS."""
    return np.array([(m, d) in HOLIDAYS for m, d in zip(index.month, index.day)], dtype=int)


def _hourly_profile(hour: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Relative load shape through the day (morning peak ~9h, evening peak ~20h)."""
    morning = np.exp(-0.5 * ((hour - 9) / 2.2) ** 2)
    evening = np.exp(-0.5 * ((hour - 20) / 2.6) ** 2)
    night_base = 0.55
    weekday_shape = night_base + 0.30 * morning + 0.45 * evening
    # weekends: later, flatter morning, same evening peak
    weekend_morning = np.exp(-0.5 * ((hour - 11) / 3.0) ** 2)
    weekend_shape = night_base + 0.18 * weekend_morning + 0.42 * evening
    return np.where(weekend == 1, weekend_shape, weekday_shape)


def simulate_energy_data(days: int = DEFAULT_DAYS, seed: int = DEFAULT_SEED,
                         start: str = START_DATE, inject_issues: bool = True) -> pd.DataFrame:
    """Return a raw hourly DataFrame: timestamp, temperature_c, humidity_pct, energy_mwh."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    weekend = (idx.dayofweek.to_numpy() >= 5).astype(int)
    holiday = is_holiday(idx)

    # ---- weather: warm-climate city. Peak heat around late May, daily peak at 15:00
    seasonal = 27 + 7 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    daily = 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    ar = np.zeros(n)                       # slow, day-to-day weather "mood" (AR(1) noise)
    shocks = rng.normal(0, 0.55, n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + shocks[i]
    temperature = seasonal + daily + ar
    humidity = np.clip(65 - 1.6 * (temperature - 27) + rng.normal(0, 5, n), 15, 100)

    # ---- demand
    base = 380.0                                           # MWh at average conditions
    profile = _hourly_profile(hour, weekend)
    day_type = np.where(holiday == 1, 0.82, np.where(weekend == 1, 0.90, 1.0))
    cooling = 7.5 * np.clip(temperature - 24, 0, None)     # AC load grows with heat
    cooling *= 1 + 0.004 * np.clip(humidity - 60, 0, None) # humid heat feels hotter
    heating = 2.0 * np.clip(16 - temperature, 0, None)
    trend = 1 + 0.03 * (np.arange(n) / (24 * 365.25))      # +3% / year growth
    noise = rng.normal(1.0, 0.025, n)
    energy = (base * profile * day_type + cooling * profile + heating) * trend * noise

    df = pd.DataFrame({TIME_COL: idx, "temperature_c": temperature.round(2),
                       "humidity_pct": humidity.round(1), TARGET: energy.round(2)})

    if inject_issues:                                      # make raw data "dirty" on purpose
        n_missing = int(0.006 * n)
        df.loc[rng.choice(n, n_missing, replace=False), TARGET] = np.nan        # sensor dropout
        spikes = rng.choice(n, int(0.002 * n), replace=False)
        df.loc[spikes, TARGET] = df.loc[spikes, TARGET] * rng.uniform(2.0, 2.8, len(spikes))
        dips = rng.choice(n, int(0.001 * n), replace=False)
        df.loc[dips, TARGET] = df.loc[dips, TARGET] * rng.uniform(0.1, 0.3, len(dips))
        df.loc[rng.choice(n, int(0.003 * n), replace=False), "temperature_c"] = np.nan
        df = df.drop(index=rng.choice(np.arange(24, n - 24), 25, replace=False))  # missing rows
        df = pd.concat([df, df.sample(5, random_state=seed)])                      # duplicates
        df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    return df
