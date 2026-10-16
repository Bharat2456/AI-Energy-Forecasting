"""Run with:  python -m unittest discover -s tests -v      (or: pytest -v, if installed)"""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.data_generator import simulate_energy_data
from src.features import FEATURE_COLUMNS, build_dataset, make_features
from src.forecasting import recursive_forecast
from src.models import load_model, regression_metrics
from src.pipeline import run_pipeline
from src.preprocessing import clean_data, load_raw
from src.simulation import procurement_cost


class TestDataGenerator(unittest.TestCase):
    def test_same_seed_same_data(self):
        a = simulate_energy_data(days=30, seed=1)
        b = simulate_energy_data(days=30, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_raw_data_is_dirty_on_purpose(self):
        raw = simulate_energy_data(days=60, seed=2)
        self.assertTrue(raw[C.TARGET].isna().any())
        self.assertTrue(raw.duplicated("timestamp").any())

    def test_evening_peak_higher_than_night(self):
        raw = simulate_energy_data(days=60, seed=3, inject_issues=False)
        by_hour = raw.groupby(pd.to_datetime(raw["timestamp"]).dt.hour)[C.TARGET].mean()
        self.assertGreater(by_hour[20], by_hour[3] * 1.4)


class TestPreprocessing(unittest.TestCase):
    def test_clean_removes_all_problems(self):
        raw = simulate_energy_data(days=60, seed=4)
        clean, rep = clean_data(raw)
        self.assertEqual(clean.isna().sum().sum(), 0)
        self.assertFalse(clean.index.duplicated().any())
        self.assertTrue((clean.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all())
        self.assertGreater(rep["duplicates_removed"], 0)
        self.assertGreater(rep["outliers_flagged"], 0)
        self.assertLess(clean[C.TARGET].max(), 700)  # sensor spikes (>1000) are gone

    def test_missing_column_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "bad.csv"
            pd.DataFrame({"timestamp": ["2024-01-01"], "x": [1]}).to_csv(f, index=False)
            with self.assertRaises(ValueError):
                load_raw(f)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        raw = simulate_energy_data(days=40, seed=5)
        self.clean, _ = clean_data(raw)

    def test_no_leakage_lags_are_past_values(self):
        f = make_features(self.clean)
        t = self.clean.index[300]
        self.assertAlmostEqual(f.loc[t, "lag_24h"], self.clean.loc[t - pd.Timedelta(hours=24), C.TARGET])
        self.assertAlmostEqual(f.loc[t, "lag_168h"], self.clean.loc[t - pd.Timedelta(hours=168), C.TARGET])
        # rolling mean only uses data >= 24h old
        past = self.clean.loc[t - pd.Timedelta(hours=47): t - pd.Timedelta(hours=24), C.TARGET]
        self.assertAlmostEqual(f.loc[t, "roll_mean_24h"], past.mean())

    def test_dataset_complete_and_aligned(self):
        X, y = build_dataset(self.clean)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertEqual(X.isna().sum().sum(), 0)
        self.assertTrue(X.index.equals(y.index))


class TestMetricsAndCost(unittest.TestCase):
    def test_perfect_prediction(self):
        m = regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(m["RMSE"], 0)
        self.assertEqual(m["R2"], 1)

    def test_rmse_value(self):
        self.assertAlmostEqual(regression_metrics([0, 0], [3, 4])["RMSE"], np.sqrt(12.5))

    def test_cost_is_lowest_for_perfect_forecast(self):
        actual = np.array([100.0, 200.0, 300.0])
        self.assertLess(procurement_cost(actual, actual), procurement_cost(actual, actual * 0.8))
        self.assertLess(procurement_cost(actual, actual), procurement_cost(actual, actual * 1.2))


class TestEndToEnd(unittest.TestCase):
    """Full pipeline on a small simulated dataset in a temporary folder (~30-60 s)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = C.Paths(cls.tmp.name)
        cls.result = run_pipeline(cls.paths, days=400, seed=7, quick=True, plots=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_exist(self):
        for rel in ["outputs/metrics.csv", "outputs/metrics.json", "outputs/forecast_next_7_days.csv",
                    "outputs/simulation_report.json", "outputs/training_log.txt", "models/best_model.joblib",
                    "images/05_actual_vs_predicted.png", "images/08_future_forecast.png"]:
            self.assertTrue((Path(self.tmp.name) / rel).exists(), rel)

    def test_best_model_beats_baseline_and_is_accurate(self):
        m = self.result["metrics"]
        best = self.result["best"]
        self.assertLess(m[best]["RMSE"], m["Seasonal Naive (lag 168h)"]["RMSE"])
        self.assertGreater(m[best]["R2"], 0.90)

    def test_forecast_is_valid(self):
        fc = self.result["forecast"]
        self.assertEqual(len(fc), C.FORECAST_HORIZON_HOURS)
        self.assertTrue((fc["forecast_mwh"] > 0).all())
        self.assertTrue(fc["forecast_mwh"].notna().all())
        self.assertTrue(fc.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all())

    def test_saved_model_reloads_and_forecasts(self):
        bundle = load_model(self.paths.models / "best_model.joblib")
        clean = pd.read_csv(self.paths.clean_csv, index_col=0, parse_dates=True)
        fc = recursive_forecast(bundle["model"], clean, 48)
        self.assertEqual(len(fc), 48)

    def test_model_saves_money(self):
        s = self.result["simulation"]
        self.assertLess(s["cost_oracle"], s["cost_model"])
        self.assertGreater(s["saving_model_vs_naive"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
