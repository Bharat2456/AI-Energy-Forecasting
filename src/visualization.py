"""All graphs. Each function saves a PNG into images/ and returns its path."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # no display needed -> works on servers, WSL, CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import TARGET

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"actual": "#1f2937", "pred": "#2563eb", "naive": "#9ca3af", "fc": "#ea580c"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_overview(clean, images):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    daily = clean[TARGET].resample("D").sum()
    ax[0].plot(daily.index, daily, color=PALETTE["actual"], lw=0.8, label="Daily total")
    ax[0].plot(daily.rolling(30, center=True).mean(), color=PALETTE["fc"], lw=2, label="30-day average")
    ax[0].set(title="Daily energy consumption (whole history)", ylabel="MWh / day"); ax[0].legend()
    last = clean[TARGET].tail(24 * 14)
    ax[1].plot(last.index, last, color=PALETTE["pred"], lw=1.2)
    ax[1].set(title="Last 14 days, hourly - note the daily double-peak", ylabel="MWh / hour")
    return _save(fig, images / "01_consumption_overview.png")


def plot_patterns(clean, images):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    pivot = clean.assign(hour=clean.index.hour, dow=clean.index.dayofweek).pivot_table(
        index="dow", columns="hour", values=TARGET, aggfunc="mean")
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax[0], cbar_kws={"label": "avg MWh"})
    ax[0].set(title="Average load: weekday x hour", xlabel="Hour of day", ylabel="")
    monthly = clean[TARGET].groupby(clean.index.month).mean()
    ax[1].bar(monthly.index, monthly.values, color=PALETTE["pred"])
    ax[1].set(title="Average load by month", xlabel="Month", ylabel="MWh / hour")
    ax[2].scatter(clean["temperature_c"], clean[TARGET], s=3, alpha=0.25, color=PALETTE["fc"])
    ax[2].set(title="Temperature vs demand (AC effect)", xlabel="Temperature (°C)", ylabel="MWh / hour")
    return _save(fig, images / "02_demand_patterns.png")


def plot_cleaning(raw, clean, images):
    """Before/after view of the cleaning step on a window that contains outliers."""
    r = raw.copy(); r["timestamp"] = pd.to_datetime(r["timestamp"]); r = r.drop_duplicates("timestamp").set_index("timestamp")
    dev = (r[TARGET] / r[TARGET].rolling(25, center=True, min_periods=5).median()).sub(1).abs()
    center = dev.idxmax() if dev.notna().any() else r.index[len(r) // 2]
    sl = slice(center - pd.Timedelta(days=4), center + pd.Timedelta(days=4))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(r.loc[sl].index, r.loc[sl, TARGET], "o", ms=3, color="#dc2626", label="Raw (with glitches/gaps)")
    ax.plot(clean.loc[sl].index, clean.loc[sl, TARGET], color=PALETTE["pred"], lw=1.4, label="Cleaned")
    ax.set(title="Preprocessing: raw vs cleaned data (window with biggest anomaly)", ylabel="MWh / hour")
    ax.legend()
    return _save(fig, images / "03_preprocessing_before_after.png")


def plot_model_comparison(metrics, images):
    df = pd.DataFrame(metrics).T
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    colors = ["#9ca3af" if "Naive" in n else PALETTE["pred"] for n in df.index]
    ax[0].barh(df.index, df["RMSE"], color=colors); ax[0].set(title="Test RMSE (lower is better)", xlabel="MWh")
    ax[1].barh(df.index, df["R2"], color=colors); ax[1].set(title="Test R² (higher is better)", xlim=(max(0, df["R2"].min() - 0.05), 1.0))
    for a in ax:
        a.invert_yaxis()
    return _save(fig, images / "04_model_comparison.png")


def plot_actual_vs_pred(y_test, pred, best, images, days=7):
    n = days * 24
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(y_test.index[-n:], y_test.iloc[-n:], color=PALETTE["actual"], lw=1.8, label="Actual")
    ax[0].plot(y_test.index[-n:], pred[best].iloc[-n:], color=PALETTE["pred"], lw=1.8, ls="--", label=f"Predicted ({best})")
    ax[0].set(title=f"Actual vs predicted - last {days} days of test set", ylabel="MWh / hour"); ax[0].legend()
    lim = [y_test.min(), y_test.max()]
    ax[1].scatter(y_test, pred[best], s=4, alpha=0.35, color=PALETTE["pred"])
    ax[1].plot(lim, lim, color="#dc2626", lw=1.2, label="Perfect prediction")
    ax[1].set(title="Actual vs predicted (every test hour)", xlabel="Actual MWh", ylabel="Predicted MWh"); ax[1].legend()
    return _save(fig, images / "05_actual_vs_predicted.png")


def plot_residuals(y_test, pred, best, images):
    res = y_test.to_numpy() - pred[best].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(res, bins=50, color=PALETTE["pred"]); ax[0].axvline(0, color="#dc2626")
    ax[0].set(title=f"Error distribution ({best})", xlabel="Actual - predicted (MWh)")
    hr = pd.Series(np.abs(res), index=y_test.index).groupby(y_test.index.hour).mean()
    ax[1].bar(hr.index, hr.values, color=PALETTE["fc"]); ax[1].set(title="Mean absolute error by hour of day", xlabel="Hour", ylabel="MAE (MWh)")
    return _save(fig, images / "06_residual_analysis.png")


def plot_importance(imp, best, images, top=15):
    imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp.index, imp.values, color=PALETTE["pred"]); ax.set(title=f"Top feature importances ({best})")
    return _save(fig, images / "07_feature_importance.png")


def plot_forecast(clean, forecast, images, lookback_days=7):
    hist = clean[TARGET].tail(lookback_days * 24)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(hist.index, hist, color=PALETTE["actual"], lw=1.5, label="History")
    ax.plot(forecast.index, forecast["forecast_mwh"], color=PALETTE["fc"], lw=1.8, label="7-day forecast")
    ax.axvline(forecast.index[0], color="#6b7280", ls=":")
    ax.set(title="Future energy demand forecast", ylabel="MWh / hour"); ax.legend()
    return _save(fig, images / "08_future_forecast.png")


def plot_cost_simulation(sim, images):
    labels = ["Oracle\n(perfect)", "ML model", "Naive\n(last week)"]
    vals = [sim["cost_oracle"], sim["cost_model"], sim["cost_naive"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    base = min(vals)
    ax.bar(labels, [v - base for v in vals], color=["#16a34a", PALETTE["pred"], "#9ca3af"])
    ax.set(title="Extra procurement cost over the perfect forecast (test period)", ylabel="Currency units")
    return _save(fig, images / "09_cost_simulation.png")
