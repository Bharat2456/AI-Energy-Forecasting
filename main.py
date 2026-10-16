"""Command-line entry point.

    python main.py                  # full pipeline (simulate -> clean -> train -> forecast -> graphs)
    python main.py --quick          # faster, lighter models (good for a first try / tests)
    python main.py --data my.csv    # use your own CSV (timestamp, energy_mwh, temperature_c, humidity_pct)
    python main.py --predict-only   # load models/best_model.joblib and just forecast again
"""
from __future__ import annotations
import argparse
import sys

import pandas as pd

from src import config as C
from src.forecasting import recursive_forecast
from src.models import load_model
from src.pipeline import run_pipeline


def main(argv=None):
    p = argparse.ArgumentParser(description="AI-Powered Energy Consumption Forecasting System")
    p.add_argument("--days", type=int, default=C.DEFAULT_DAYS, help="days of data to simulate (default 1096)")
    p.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    p.add_argument("--quick", action="store_true", help="smaller models, finishes in seconds")
    p.add_argument("--data", help="path to your own CSV instead of simulated data")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--predict-only", action="store_true", help="reuse saved model + cleaned data")
    p.add_argument("--horizon", type=int, default=C.FORECAST_HORIZON_HOURS, help="hours to forecast (predict-only)")
    a = p.parse_args(argv)

    paths = C.Paths().make()
    if a.predict_only:
        if not (paths.models / "best_model.joblib").exists() or not paths.clean_csv.exists():
            sys.exit("No saved model/data found. Run `python main.py` first.")
        bundle = load_model(paths.models / "best_model.joblib")
        clean = pd.read_csv(paths.clean_csv, index_col=0, parse_dates=True)
        fc = recursive_forecast(bundle["model"], clean, a.horizon)
        fc.to_csv(paths.outputs / "forecast_next_7_days.csv")
        print(f"Model: {bundle['name']}\n{fc.head(24).to_string()}\n... saved to outputs/forecast_next_7_days.csv")
        return
    run_pipeline(paths, days=a.days, seed=a.seed, quick=a.quick, data_csv=a.data, plots=not a.no_plots)


if __name__ == "__main__":
    main()
