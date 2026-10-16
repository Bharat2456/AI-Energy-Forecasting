# 08 · Web dashboard (UI / UX layer)

The original project is a command-line pipeline. The **web dashboard** is an optional layer on top of it so that
anybody can use the system **without a terminal**. Nothing in the pipeline changed: the dashboard reads the same
files (`outputs/`, `data/`, `models/`, `images/`) and starts the same `run_pipeline()` that `python main.py` uses.

## Start it
| OS | How |
|---|---|
| Windows | double-click `run_app_windows.bat` |
| macOS / Linux | `bash run_app_mac_linux.sh` |
| Manual | `pip install -r requirements.txt` then `python app.py` |

The browser opens at <http://127.0.0.1:5000>. Options: `python app.py --port 8000`, `python app.py --no-browser`.
It is **localhost-only** (bound to 127.0.0.1), has **no login and no database**, and needs **no internet** (charts are drawn
with plain HTML canvas, no CDN).

## Pages
| Page | What it gives you |
|---|---|
| Dashboard | KPIs (best model, accuracy, error, money saved), history + 7-day forecast chart, hourly / monthly demand patterns |
| Forecast | Interactive hourly forecast, horizon 24 h – 14 days, **temperature what-if slider** (heat-wave / cold-spell), peak-alert days, daily table, CSV download |
| Models | Test-set metrics table, RMSE bars, feature importance, 7-day backtest, data-split explanation |
| Business impact | Cost of the oracle / ML / naive planners and the money saved, plus the price assumptions |
| Data & quality | What cleaning fixed, raw-vs-clean rows, demand by weekday |
| Graph gallery | All 9 PNG graphs with click-to-enlarge |
| Run / Retrain | Train from the browser: simulated data or **upload your own CSV**, Quick/Full mode, live log + progress bar |

## Architecture
```
browser (webapp/static/index.html)  ⇄  Flask JSON API (webapp/server.py)  ⇄  src/ (unchanged pipeline) + files
```
Training runs in a background thread; the page polls `/api/run/status` and shows `outputs/training_log.txt` live.

## Notes
* Running training from the UI overwrites the generated files, exactly like `python main.py`.
* The temperature what-if replaces the seasonal-average weather used by the recursive forecast with the same average
  shifted by N °C, so you can see how a heat-wave changes the peak and the cost.
* If the saved model was trained with another scikit-learn version and fails to load, press **Run / Retrain**.
