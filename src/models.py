"""Model training, selection and evaluation.

Models compared (all from scikit-learn, no GPU needed):
* Seasonal Naive  - baseline: "demand = same hour last week"
* Linear Regression (with scaling)
* Random Forest
* Gradient Boosting (HistGradientBoostingRegressor)

Split is *chronological* (never shuffle time series!):
    train | validation (60 d) | test (90 d)
The validation block picks the winner; the test block gives the final honest score.
"""
from __future__ import annotations
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_SEED, TEST_DAYS, VAL_DAYS
from .features import FEATURE_COLUMNS

log = logging.getLogger("energy")
BASELINE = "Seasonal Naive (lag 168h)"


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = y_true != 0
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else float("nan"),
        "R2": float(r2_score(y_true, y_pred)),
    }


def chronological_split(X, y, val_days=VAL_DAYS, test_days=TEST_DAYS):
    n_test, n_val = test_days * 24, val_days * 24
    if len(X) < (n_test + n_val) * 2:
        raise ValueError("Not enough data: need at least ~2x (validation + test) hours of history.")
    t0, t1 = len(X) - n_test - n_val, len(X) - n_test
    return (X.iloc[:t0], y.iloc[:t0]), (X.iloc[t0:t1], y.iloc[t0:t1]), (X.iloc[t1:], y.iloc[t1:])


def get_models(seed=DEFAULT_SEED, quick=False):
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=60 if quick else 200, max_depth=16,
                                               min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=150 if quick else 400,
                                                           learning_rate=0.06, random_state=seed),
    }


def train_and_evaluate(X, y, seed=DEFAULT_SEED, quick=False):
    """Fit every model on train, score on validation + test. Returns dict with everything."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(X, y)
    log.info("Rows -> train: %d | validation: %d | test: %d", len(Xtr), len(Xva), len(Xte))
    log.info("Train period: %s -> %s", Xtr.index.min(), Xtr.index.max())
    log.info("Test  period: %s -> %s", Xte.index.min(), Xte.index.max())

    results, val_scores, test_pred, fitted = {}, {}, {}, {}
    # baseline needs no training
    val_scores[BASELINE] = regression_metrics(yva, Xva["lag_168h"])
    results[BASELINE] = regression_metrics(yte, Xte["lag_168h"])
    test_pred[BASELINE] = Xte["lag_168h"].to_numpy()
    log.info("%-28s test RMSE=%.2f  R2=%.4f", BASELINE, results[BASELINE]["RMSE"], results[BASELINE]["R2"])

    for name, model in get_models(seed, quick).items():
        model.fit(Xtr, ytr)
        val_scores[name] = regression_metrics(yva, model.predict(Xva))
        pred = model.predict(Xte)
        results[name] = regression_metrics(yte, pred)
        test_pred[name] = pred
        fitted[name] = model
        log.info("%-28s val RMSE=%.2f | test RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%  R2=%.4f", name,
                 val_scores[name]["RMSE"], results[name]["RMSE"], results[name]["MAE"],
                 results[name]["MAPE_%"], results[name]["R2"])

    best = min(fitted, key=lambda k: val_scores[k]["RMSE"])
    log.info("Best model on validation RMSE: %s", best)
    return {"metrics_test": results, "metrics_val": val_scores, "best": best, "fitted": fitted,
            "test_pred": pd.DataFrame(test_pred, index=Xte.index), "y_test": yte,
            "split": {"train_end": str(Xtr.index.max()), "val_end": str(Xva.index.max()),
                      "test_start": str(Xte.index.min()), "test_end": str(Xte.index.max())}}


def refit_on_all(name, X, y, seed=DEFAULT_SEED, quick=False):
    """After selection, retrain the winner on ALL history so forecasts use the newest data."""
    model = get_models(seed, quick)[name]
    model.fit(X, y)
    return model


def feature_importance(model, X=None, y=None) -> pd.Series | None:
    """Random Forest: built-in importances. Linear Regression: |standardised coefficients|.
    Anything else (e.g. HistGradientBoosting): permutation importance on the last 30 days."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    if hasattr(model, "named_steps"):
        coef = np.abs(model.named_steps["linearregression"].coef_)
        return pd.Series(coef / coef.sum(), index=FEATURE_COLUMNS).sort_values(ascending=False)
    if X is not None:
        from sklearn.inspection import permutation_importance
        Xs, ys = X.tail(24 * 30), y.tail(24 * 30)
        r = permutation_importance(model, Xs, ys, n_repeats=3, random_state=0, scoring="neg_root_mean_squared_error")
        imp = pd.Series(np.clip(r.importances_mean, 0, None), index=FEATURE_COLUMNS)
        return (imp / imp.sum()).sort_values(ascending=False)
    return None


def save_model(model, name, metrics, path):
    joblib.dump({"model": model, "name": name, "feature_columns": FEATURE_COLUMNS,
                 "test_metrics": metrics}, path)


def load_model(path):
    return joblib.load(path)
