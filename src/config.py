"""Central configuration: folder paths, constants and tariff assumptions.

Everything a beginner may want to tweak lives here, so the other modules stay clean.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- data columns
TIME_COL = "timestamp"
TARGET = "energy_mwh"          # electricity demand of the simulated city district (MWh per hour)
WEATHER_COLS = ["temperature_c", "humidity_pct"]

# ---------------------------------------------------------------- simulation
START_DATE = "2022-01-01"
DEFAULT_DAYS = 1096            # 3 years of hourly data (2022-2024)
DEFAULT_SEED = 42
# Fixed-date public holidays (month, day) used by the simulator and feature engineering.
HOLIDAYS = {(1, 1), (1, 26), (3, 8), (8, 15), (10, 2), (11, 1), (12, 25)}

# ---------------------------------------------------------------- features
LAGS = [24, 48, 72, 168]       # hours. All >= 24 => "day-ahead" forecasting, no leakage.
ROLL_WINDOWS = [24, 168]
FORECAST_MIN_LAG = 24

# ---------------------------------------------------------------- modelling
TEST_DAYS = 90                 # last 90 days = final test set
VAL_DAYS = 60                  # 60 days before the test set = validation (model selection)
FORECAST_HORIZON_HOURS = 168   # 7 days

# ---------------------------------------------------------------- business simulation
PRICE_DAY_AHEAD = 5000.0       # currency units per MWh bought day-ahead (e.g. INR 5/kWh)
SHORTFALL_MULT = 1.8           # under-forecast: missing energy bought on spot market at 1.8x
SURPLUS_MULT = 0.5             # over-forecast: unused energy resold at only 0.5x
PEAK_QUANTILE = 0.95           # top 5% hours are "peak" hours
PEAK_SHIFT_FRACTION = 0.05     # demand response: shift 5% of peak load away from peak hours


@dataclass
class Paths:
    """All project folders, relative to a root (tests pass a temp folder as root)."""
    root: Path = BASE_DIR

    def __post_init__(self):
        self.root = Path(self.root)
        self.data = self.root / "data"
        self.models = self.root / "models"
        self.outputs = self.root / "outputs"
        self.images = self.root / "images"

    def make(self):
        for p in (self.data, self.models, self.outputs, self.images):
            p.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def raw_csv(self):
        return self.data / "energy_raw.csv"

    @property
    def clean_csv(self):
        return self.data / "energy_clean.csv"
