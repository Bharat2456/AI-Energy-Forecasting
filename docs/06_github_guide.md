# 6. GitHub upload strategy

**Repository name:** `AI-Energy-Consumption-Forecasting`
**Description:** `End-to-end ML pipeline that simulates smart-city electricity demand, forecasts 7 days ahead (R² 0.96) and quantifies business savings. Python, scikit-learn.`
**Topics/tags:** `machine-learning` `time-series-forecasting` `energy-forecasting` `data-science` `python` `scikit-learn` `smart-grid` `data-visualization` `portfolio-project` `simulation`

## Create & push
```bash
# 1. On github.com: New repository -> name above -> Public -> (do NOT add README/.gitignore; you have them)
cd AI-Energy-Forecasting
git init
git branch -M main
git config user.name "Your Name"; git config user.email "you@example.com"
git add .
git commit -m "feat: initial project structure"
git remote add origin https://github.com/<your-username>/AI-Energy-Consumption-Forecasting.git
git push -u origin main
```
## Make it professional
* Keep `images/` and `outputs/` committed – recruiters see results without running code.
* Pin the repo on your profile; add the description + topics (⚙ next to *About*).
* Add a `LICENSE` (Add file → Create new file → `LICENSE` → choose MIT).
* Add a repo social preview (Settings → Social preview) using `images/05_actual_vs_predicted.png`.
* Tag a release: `git tag v1.0 && git push --tags`.
* Put a one-line CV bullet: *"Built an ML energy-demand forecasting pipeline (Gradient Boosting, R² 0.96, 2.9 % MAPE) with leakage-free features, 7-day recursive forecasting and a cost-saving simulation."*

## 7-day proof plan (commit per day)
| Day | Work | Commit message | Proof |
|---|---|---|---|
| 1 | Setup: repo, venv, folders, `requirements.txt`, `.gitignore`, `config.py` | `chore: project scaffold and environment setup` | terminal screenshot of `pip install` |
| 2 | Dataset: `data_generator.py`, raw CSV, preview | `feat: simulate hourly smart-city energy data` | `data_preview.txt` |
| 3 | Preprocessing: `preprocessing.py`, report, before/after graph | `feat: data cleaning with outlier and gap handling` | `preprocessing_report.json`, graph 03 |
| 4 | Model: `features.py`, `models.py` | `feat: leakage-free features and model training` | `training_log.txt` |
| 5 | Evaluation: metrics, backtest, tests | `feat: evaluation metrics, 7-day backtest and unit tests` | `metrics.csv`, passing tests |
| 6 | Visualization + simulation: `visualization.py`, `simulation.py`, forecast | `feat: forecasting, graphs and cost simulation` | graphs 05-09 |
| 7 | Upload: README, docs, screenshots, tag `v1.0` | `docs: README with results and usage` | repo page screenshot |

Tips: small, meaningful commits; imperative messages (`feat:`, `fix:`, `docs:`, `test:`); never commit `.venv`.
