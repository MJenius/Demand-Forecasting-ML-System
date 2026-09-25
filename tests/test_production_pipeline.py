from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from production_pipeline import baseline_predictions, build_features, drift_report, full_run_memory_preflight, metrics, promotion_decision, psi, validate_data


def small_raw():
    sales = pd.DataFrame({
        "id": ["x"], "item_id": ["i"], "dept_id": ["d"], "cat_id": ["c"], "store_id": ["s"], "state_id": ["CA"],
        **{f"d_{day}": [float(day)] for day in range(1, 81)},
    })
    dates = pd.date_range("2020-01-01", periods=90)
    calendar = pd.DataFrame({
        "date": dates.astype(str), "d": [f"d_{day}" for day in range(1, 91)], "wm_yr_wk": np.repeat(np.arange(13), 7)[:90],
        "event_name_1": None, "event_name_2": None, "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
    })
    prices = pd.DataFrame({"store_id": "s", "item_id": "i", "wm_yr_wk": np.arange(13), "sell_price": 1.0})
    return sales, calendar, prices


def test_horizon_features_cannot_see_future_targets():
    sales, calendar, prices = small_raw()
    features = build_features(sales, calendar, prices, horizon=7)
    row = features.loc[features["date"] == pd.Timestamp("2020-01-20")].iloc[0]
    assert row["lag_7"] == 13.0
    assert row["rolling_mean_7"] == pytest.approx(np.mean(range(7, 14)))
    changed = sales.copy()
    changed.loc[0, [f"d_{day}" for day in range(14, 21)]] = 9999
    changed_row = build_features(changed, calendar, prices, 7).loc[lambda x: x["date"] == "2020-01-20"].iloc[0]
    assert changed_row["lag_7"] == row["lag_7"]
    assert changed_row["rolling_mean_7"] == row["rolling_mean_7"]


def test_data_quality_rejects_invalid_values(tmp_path):
    sales, calendar, prices = small_raw()
    sales.loc[0, "d_3"] = -1
    with pytest.raises(ValueError, match="negative demand"):
        validate_data(sales, calendar, prices, {"data": {"max_calendar_gap_days": 60}}, tmp_path / "schema.json")


def test_data_quality_detects_schema_change(tmp_path):
    sales, calendar, prices = small_raw()
    config = {"data": {"max_calendar_gap_days": 60}}
    validate_data(sales, calendar, prices, config, tmp_path / "schema.json")
    with pytest.raises(ValueError, match="schema changed"):
        validate_data(sales.assign(extra=1), calendar, prices, config, tmp_path / "schema.json")


def test_psi_detects_shift_and_handles_constant_values():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 10000), rng.normal(0, 1, 10000)) < 0.1
    assert psi(rng.normal(0, 1, 10000), rng.normal(3, 1, 10000)) > 0.25
    assert psi(np.ones(20), np.ones(20)) == 0
    assert np.isinf(psi(np.ones(20), np.zeros(20)))


def test_promotion_rejects_any_inferior_guardrail_metric():
    current = {"MAE": 2.0, "RMSE": 3.0, "WAPE": 0.40, "Bias": 0.05}
    assert promotion_decision(current, {"MAE": 1.9, "RMSE": 2.9, "WAPE": 0.38, "Bias": 0.02})["decision"] == "PROMOTE"
    assert promotion_decision(current, {"MAE": 1.9, "RMSE": 3.1, "WAPE": 0.38, "Bias": 0.02})["decision"] == "REJECT"
    assert promotion_decision(current, {"MAE": 2.1, "RMSE": 2.9, "WAPE": 0.38, "Bias": 0.02})["decision"] == "REJECT"



def test_metrics_are_measured_and_zero_safe():
    result = metrics([0, 2, 4], [1, 1, 5])
    assert result["MAE"] == 1
    assert result["RMSE"] == 1
    assert result["MAPE"] == pytest.approx(37.5)


def test_baselines_only_use_information_available_at_origin():
    sales, calendar, prices = small_raw()
    featured = build_features(sales, calendar, prices, 7)
    train = featured[featured["date"] <= "2020-01-19"]
    validation = featured[featured["date"] == "2020-01-20"]
    predictions = baseline_predictions(train, validation, 7)
    assert predictions["seasonal_naive"][0] == 13
    assert predictions["moving_average"][0] == 7


def test_drift_report_uses_two_full_windows():
    dates = np.repeat(pd.date_range("2020-01-01", periods=56), 10)
    values = np.r_[np.zeros(280), np.full(280, 3.0)]
    frame = pd.DataFrame({"date": dates, "x": values, "target": values})

    class IdentityModel:
        def predict(self, data):
            return data["x"].to_numpy()

    report = drift_report(frame, IdentityModel(), ["x"], {"drift": {"bins": 10, "warning_psi": 0.1, "critical_psi": 0.25}})
    assert report["status"] == "DRIFT"
    assert report["feature_psi"]["x"] > 0.25


def test_full_run_preflight_fails_before_memory_exhaustion(monkeypatch):
    monkeypatch.setattr("production_pipeline.psutil.virtual_memory", lambda: SimpleNamespace(available=1 * 1024**3))
    with pytest.raises(MemoryError, match="1.6 GB"):
        full_run_memory_preflight({"data": {"sample_items": 3049, "minimum_available_memory_gb": 1.6}})
