"""End-to-end pipeline used by main.py and by the tests."""
from __future__ import annotations
import json
import logging
import time

import pandas as pd

from . import config as C
from .data_generator import simulate_energy_data
from .features import build_dataset
from .forecasting import recursive_backtest, recursive_forecast
from .models import (BASELINE, feature_importance, refit_on_all, save_model,
                     train_and_evaluate)
from .preprocessing import clean_data, load_raw
from .simulation import peak_report, run_cost_simulation
from . import visualization as viz

log = logging.getLogger("energy")


def setup_logging(paths):
    paths.make()
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(paths.outputs / "training_log.txt", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _run_pipeline(paths=None, days=C.DEFAULT_DAYS, seed=C.DEFAULT_SEED, quick=False,
                 data_csv=None, plots=True):
    paths = (paths or C.Paths()).make()
    setup_logging(paths)
    t0 = time.time()

    # 1) DATA --------------------------------------------------------------
    if data_csv is None:
        log.info("[1/7] Simulating %d days of hourly smart-city energy data (seed=%d)", days, seed)
        simulate_energy_data(days=days, seed=seed).to_csv(paths.raw_csv, index=False)
        src = paths.raw_csv
    else:
        log.info("[1/7] Using your dataset: %s", data_csv)
        src = data_csv
    raw = load_raw(src)
    (paths.outputs / "data_preview.txt").write_text(
        "RAW DATA - first 10 rows\n" + raw.head(10).to_string(index=False) +
        "\n\nSummary statistics\n" + raw.describe().round(2).to_string())

    # 2) PREPROCESSING -----------------------------------------------------
    log.info("[2/7] Cleaning data")
    clean, report = clean_data(raw)
    clean.to_csv(paths.clean_csv)
    _dump(report, paths.outputs / "preprocessing_report.json")
    log.info("      %s", report)

    # 3) FEATURES ----------------------------------------------------------
    log.info("[3/7] Engineering features")
    X, y = build_dataset(clean)
    log.info("      feature matrix: %d rows x %d columns", *X.shape)

    # 4+5) TRAIN / EVALUATE ------------------------------------------------
    log.info("[4/7] Training and evaluating models (quick=%s)", quick)
    res = train_and_evaluate(X, y, seed=seed, quick=quick)
    best = res["best"]
    metrics = pd.DataFrame(res["metrics_test"]).T.round(4)
    metrics.index.name = "model"
    metrics.to_csv(paths.outputs / "metrics.csv")
    _dump({"best_model": best, "split": res["split"], "test": res["metrics_test"],
           "validation": res["metrics_val"]}, paths.outputs / "metrics.json")
    log.info("\n%s", metrics.to_string())

    # 6) FORECAST ----------------------------------------------------------
    log.info("[5/7] Re-training '%s' on all data and forecasting next 7 days", best)
    final = refit_on_all(best, X, y, seed=seed, quick=quick)
    save_model(final, best, res["metrics_test"][best], paths.models / "best_model.joblib")
    forecast = recursive_forecast(final, clean, C.FORECAST_HORIZON_HOURS)
    forecast.to_csv(paths.outputs / "forecast_next_7_days.csv")
    backtest = recursive_backtest(res["fitted"][best], clean, res["split"]["test_start"], res["split"]["test_end"])
    _dump(backtest, paths.outputs / "backtest_7day.json")
    log.info("      7-day recursive backtest RMSE per window: %s", [round(b["RMSE"], 2) for b in backtest])

    # 7) SIMULATION --------------------------------------------------------
    log.info("[6/7] Business simulation (procurement cost + peak planning)")
    sim = run_cost_simulation(res["y_test"], res["test_pred"][best], res["test_pred"][BASELINE], best)
    sim["peak_planning"] = peak_report(forecast, clean)
    _dump(sim, paths.outputs / "simulation_report.json")
    log.info("      cost saving vs naive planner: %.0f (%.1f%% of total cost)",
             sim["saving_model_vs_naive"], sim["saving_pct_of_total"])

    # 8) PLOTS -------------------------------------------------------------
    if plots:
        log.info("[7/7] Drawing graphs into %s", paths.images)
        viz.plot_overview(clean, paths.images); viz.plot_patterns(clean, paths.images)
        viz.plot_cleaning(raw, clean, paths.images)
        viz.plot_model_comparison(res["metrics_test"], paths.images)
        viz.plot_actual_vs_pred(res["y_test"], res["test_pred"], best, paths.images)
        viz.plot_residuals(res["y_test"], res["test_pred"], best, paths.images)
        imp = feature_importance(final, X, y)
        if imp is not None:
            viz.plot_importance(imp, best, paths.images)
        viz.plot_forecast(clean, forecast, paths.images)
        viz.plot_cost_simulation(sim, paths.images)
    log.info("Done in %.1fs. Best model: %s (test RMSE %.2f MWh, R2 %.4f)", time.time() - t0, best,
             res["metrics_test"][best]["RMSE"], res["metrics_test"][best]["R2"])
    return {"best": best, "metrics": res["metrics_test"], "forecast": forecast, "simulation": sim,
            "backtest": backtest, "report": report}


def close_logging():
    """Release the log file. Windows cannot delete a file that is still open."""
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


def run_pipeline(*args, **kwargs):
    """Run the whole pipeline (see _run_pipeline) and always close the log file afterwards."""
    try:
        return _run_pipeline(*args, **kwargs)
    finally:
        close_logging()
