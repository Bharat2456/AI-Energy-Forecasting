"""Virtual business simulation: what is a better forecast worth?

An electricity retailer / smart-city utility must buy energy for tomorrow *today*
(day-ahead market). Forecast errors are expensive:
  * under-forecast -> missing energy is bought on the spot market at 1.8x price
  * over-forecast  -> unused energy is resold at only 0.5x price
We replay the test period with three "planners" and compare their total cost:
  1. Oracle (perfect forecast)  - theoretical lower bound
  2. ML model forecast
  3. Naive planner (same hour last week)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def procurement_cost(actual, forecast, price=C.PRICE_DAY_AHEAD,
                     shortfall_mult=C.SHORTFALL_MULT, surplus_mult=C.SURPLUS_MULT) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    shortfall = np.clip(actual - forecast, 0, None)
    surplus = np.clip(forecast - actual, 0, None)
    return float(np.sum(price * forecast + shortfall * price * shortfall_mult - surplus * price * surplus_mult))


def run_cost_simulation(y_test: pd.Series, model_pred, naive_pred, model_name: str) -> dict:
    oracle = procurement_cost(y_test, y_test)
    model = procurement_cost(y_test, model_pred)
    naive = procurement_cost(y_test, naive_pred)
    return {
        "period": f"{y_test.index.min()} -> {y_test.index.max()}",
        "assumptions": {"day_ahead_price_per_MWh": C.PRICE_DAY_AHEAD,
                        "shortfall_multiplier": C.SHORTFALL_MULT, "surplus_multiplier": C.SURPLUS_MULT},
        "model_used": model_name,
        "cost_oracle": oracle, "cost_model": model, "cost_naive": naive,
        "excess_cost_model_vs_oracle": model - oracle,
        "excess_cost_naive_vs_oracle": naive - oracle,
        "saving_model_vs_naive": naive - model,
        "saving_pct_of_excess": float((naive - model) / (naive - oracle) * 100) if naive != oracle else 0.0,
        "saving_pct_of_total": float((naive - model) / naive * 100),
    }


def peak_report(forecast: pd.DataFrame, history: pd.DataFrame) -> dict:
    """Peak-load planning for the forecast horizon + a demand-response what-if."""
    threshold = float(history[C.TARGET].quantile(C.PEAK_QUANTILE))
    f = forecast["forecast_mwh"]
    daily = f.groupby(f.index.date).agg(["max", "idxmax", "sum"])
    days = [{"date": str(d), "peak_mwh": round(float(r["max"]), 1),
             "peak_hour": pd.Timestamp(r["idxmax"]).strftime("%H:%M"),
             "total_mwh": round(float(r["sum"]), 1),
             "high_alert": bool(r["max"] >= threshold)} for d, r in daily.iterrows()]
    peak_mask = f >= threshold
    shifted = float((f[peak_mask] * C.PEAK_SHIFT_FRACTION).sum())
    return {"peak_threshold_mwh_hist_p95": round(threshold, 1), "daily": days,
            "hours_above_threshold": int(peak_mask.sum()),
            "demand_response_shift_mwh": round(shifted, 1),
            "forecast_max_mwh": round(float(f.max()), 1), "forecast_max_time": str(f.idxmax())}
