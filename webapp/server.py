"""Flask backend for the local dashboard.

It is a thin layer on top of the existing project: it *reads* the files that
`python main.py` produces (outputs/, data/, models/, images/) and can also *start*
the very same pipeline from the browser. No login, no database - localhost only.
"""
from __future__ import annotations
import io
import json
import threading
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from src import config as C
from src.forecasting import climatology_weather, recursive_forecast
from src.models import feature_importance, load_model
from src.pipeline import run_pipeline
from src.simulation import peak_report

warnings.filterwarnings("ignore")
STATIC = Path(__file__).parent / "static"
PATHS = C.Paths().make()
UPLOAD = PATHS.data / "uploaded_data.csv"
REQUIRED_COLS = [C.TIME_COL, C.TARGET, *C.WEATHER_COLS]

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

_cache: dict = {}
_lock = threading.Lock()
JOB = {"running": False, "started": None, "finished": None, "error": None, "params": None}


# ------------------------------------------------------------------ helpers
def _json_file(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _mtime(path: Path):
    return path.stat().st_mtime if path.exists() else None


def _cached(key, files, loader):
    """Re-load only when one of the underlying files changed (e.g. after a re-train)."""
    stamp = tuple(_mtime(f) for f in files)
    hit = _cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    val = loader()
    _cache[key] = (stamp, val)
    return val


def get_clean() -> pd.DataFrame | None:
    if not PATHS.clean_csv.exists():
        return None
    return _cached("clean", [PATHS.clean_csv],
                   lambda: pd.read_csv(PATHS.clean_csv, index_col=0, parse_dates=True))


def get_bundle():
    f = PATHS.models / "best_model.joblib"
    if not f.exists():
        return None
    return _cached("model", [f], lambda: load_model(f))


def ready() -> bool:
    return get_clean() is not None and (PATHS.models / "best_model.joblib").exists()


def _series(s: pd.Series, nd=2):
    return [round(float(v), nd) for v in s]


def _tail_log(n=200):
    f = PATHS.outputs / "training_log.txt"
    if not f.exists():
        return []
    try:
        return f.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.get("/images/<path:name>")
def images(name):
    return send_from_directory(PATHS.images, name)


# ------------------------------------------------------------------ API
@app.get("/api/status")
def status():
    return jsonify(ready=ready(), has_upload=UPLOAD.exists(), job=JOB,
                   images=sorted(p.name for p in PATHS.images.glob("*.png")))


@app.get("/api/summary")
def summary():
    metrics = _json_file(PATHS.outputs / "metrics.json")
    if metrics is None:
        return jsonify(ready=False)
    sim = _json_file(PATHS.outputs / "simulation_report.json") or {}
    sim.pop("peak_planning", None)
    clean = get_clean()
    best = metrics["best_model"]
    out = {
        "ready": ready(), "best": best, "metrics": metrics["test"], "validation": metrics["validation"],
        "split": metrics["split"], "baseline": next(k for k in metrics["test"] if k.startswith("Seasonal")),
        "preprocessing": _json_file(PATHS.outputs / "preprocessing_report.json"),
        "backtest": _json_file(PATHS.outputs / "backtest_7day.json"),
        "simulation": sim,
    }
    if clean is not None:
        t = clean[C.TARGET]
        out["data"] = {"start": str(clean.index.min()), "end": str(clean.index.max()), "rows": len(clean),
                       "mean": round(float(t.mean()), 1), "max": round(float(t.max()), 1),
                       "min": round(float(t.min()), 1), "p95": round(float(t.quantile(C.PEAK_QUANTILE)), 1),
                       "last_24h_mean": round(float(t.tail(24).mean()), 1)}
    return jsonify(out)


@app.get("/api/history")
def history():
    clean = get_clean()
    if clean is None:
        return jsonify(error="No data yet"), 404
    days = max(1, min(int(request.args.get("days", 30)), 3650))
    h = clean.tail(days * 24)
    if len(h) > 4000:                      # daily-mean downsampling keeps long ranges fast
        h = h[[C.TARGET, "temperature_c"]].resample("D").mean()
    return jsonify(t=[str(i) for i in h.index], energy=_series(h[C.TARGET]),
                   temp=_series(h["temperature_c"]))


@app.get("/api/patterns")
def patterns():
    clean = get_clean()
    if clean is None:
        return jsonify(error="No data yet"), 404
    t = clean[C.TARGET]
    hourly = t.groupby(clean.index.hour).mean()
    dow = t.groupby(clean.index.dayofweek).mean()
    month = t.groupby(clean.index.month).mean()
    return jsonify(hourly=_series(hourly), dow=_series(dow), month=_series(month))


def _forecast(horizon: int, offset: float):
    clean, bundle = get_clean(), get_bundle()
    idx = pd.date_range(clean.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    fw = climatology_weather(clean, idx)
    fw["temperature_c"] = fw["temperature_c"] + offset
    return recursive_forecast(bundle["model"], clean, horizon, fw), clean, bundle


@app.get("/api/forecast")
def forecast():
    if not ready():
        return jsonify(error="Train the model first"), 404
    horizon = max(24, min(int(request.args.get("horizon", 168)), 336))
    offset = max(-10.0, min(float(request.args.get("temp_offset", 0)), 15.0))
    try:
        fc, clean, bundle = _forecast(horizon, offset)
    except Exception as e:                                   # e.g. library-version mismatch
        return jsonify(error=f"Could not forecast: {e}. Re-train from the 'Run / Retrain' page."), 500
    pk = peak_report(fc, clean)
    total = float(fc["forecast_mwh"].sum())
    return jsonify(model=bundle["name"], horizon=horizon, temp_offset=offset,
                   t=[str(i) for i in fc.index], forecast=_series(fc["forecast_mwh"]),
                   temp=_series(fc["temperature_c"]), peak=pk, total_mwh=round(total, 1),
                   avg_mwh=round(total / horizon, 1), est_cost=round(total * C.PRICE_DAY_AHEAD),
                   last_actual_t=[str(i) for i in clean.tail(72).index],
                   last_actual=_series(clean[C.TARGET].tail(72)))


@app.get("/api/forecast.csv")
def forecast_csv():
    if not ready():
        return jsonify(error="Train the model first"), 404
    horizon = max(24, min(int(request.args.get("horizon", 168)), 336))
    offset = float(request.args.get("temp_offset", 0))
    fc, _, _ = _forecast(horizon, offset)
    buf = io.StringIO()
    fc.to_csv(buf)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=energy_forecast.csv"})


@app.get("/api/importance")
def importance():
    bundle, clean = get_bundle(), get_clean()
    if bundle is None:
        return jsonify(error="No model"), 404

    def calc():
        from src.features import build_dataset
        X, y = build_dataset(clean)
        imp = feature_importance(bundle["model"], X, y)
        return {k: round(float(v), 4) for k, v in imp.head(12).items()}
    return jsonify(_cached("imp", [PATHS.models / "best_model.joblib"], calc))


@app.get("/api/sample")
def sample():
    """First rows of raw vs. clean data + preview text for the 'Data & Quality' page."""
    raw = pd.read_csv(PATHS.raw_csv).head(12) if PATHS.raw_csv.exists() else None
    clean = get_clean()
    return jsonify(raw=None if raw is None else json.loads(raw.to_json(orient="split")),
                   clean=None if clean is None else json.loads(clean.head(12).reset_index().to_json(orient="split")))


# ------------------------------------------------------------------ upload + train
@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file received"), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify(error=f"Could not read the CSV: {e}"), 400
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return jsonify(error=f"Missing column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED_COLS)}"), 400
    df.to_csv(UPLOAD, index=False)
    ts = pd.to_datetime(df[C.TIME_COL], errors="coerce")
    return jsonify(ok=True, rows=len(df), start=str(ts.min()), end=str(ts.max()),
                   days=int((ts.max() - ts.min()).days))


@app.delete("/api/upload")
def remove_upload():
    UPLOAD.unlink(missing_ok=True)
    return jsonify(ok=True)


def _worker(params):
    try:
        run_pipeline(PATHS, days=params["days"], seed=params["seed"], quick=params["quick"],
                     data_csv=str(UPLOAD) if params["use_upload"] else None, plots=True)
        JOB["error"] = None
    except Exception as e:
        JOB["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        _cache.clear()
        JOB.update(running=False, finished=time.time())


@app.post("/api/run")
def run():
    with _lock:
        if JOB["running"]:
            return jsonify(error="A run is already in progress"), 409
        b = request.get_json(silent=True) or {}
        params = {"quick": bool(b.get("quick", True)), "days": int(b.get("days", C.DEFAULT_DAYS)),
                  "seed": int(b.get("seed", C.DEFAULT_SEED)),
                  "use_upload": bool(b.get("use_upload")) and UPLOAD.exists()}
        JOB.update(running=True, started=time.time(), finished=None, error=None, params=params)
    threading.Thread(target=_worker, args=(params,), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/run/status")
def run_status():
    return jsonify(job=JOB, log=_tail_log(), elapsed=round(time.time() - JOB["started"], 1) if JOB["started"] and JOB["running"] else None)
