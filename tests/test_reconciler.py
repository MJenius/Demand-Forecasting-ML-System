import sqlite3
import pandas as pd
import pytest

from monitoring.collect_predictions import PredictionCollector
from src.monitoring.reconciler import ActualsReconciler


def test_delayed_actuals_reconciler(tmp_path):
    db_file = tmp_path / "test_predictions.db"
    collector = PredictionCollector(db_path=str(db_file))

    # Add predictions without actuals
    collector.add_prediction(
        timestamp="2026-04-01T10:00:00",
        item_id="item_A",
        store_id="store_1",
        date="2026-04-01",
        prediction=10.0,
        model_used="lightgbm",
        model_version="v1",
    )
    collector.add_prediction(
        timestamp="2026-04-02T10:00:00",
        item_id="item_A",
        store_id="store_1",
        date="2026-04-02",
        prediction=15.0,
        model_used="lightgbm",
        model_version="v1",
    )

    reconciler = ActualsReconciler(db_path=db_file)

    # Performance before reconciliation
    perf_before = reconciler.compute_operational_performance(model_version="v1")
    assert perf_before["status"] == "NO_RECONCILED_ACTUALS"

    # Provide delayed actuals
    actuals = pd.DataFrame({
        "item_id": ["item_A", "item_A"],
        "store_id": ["store_1", "store_1"],
        "date": ["2026-04-01", "2026-04-02"],
        "sales": [12.0, 14.0],
    })

    res = reconciler.reconcile_from_dataframe(actuals)
    assert res["updated_records"] == 2

    # Performance after reconciliation
    perf = reconciler.compute_operational_performance(model_version="v1")
    assert perf["status"] == "OK"
    assert perf["count"] == 2
    # Errors: |12-10| = 2, |14-15| = 1 => MAE = 1.5
    assert perf["MAE"] == pytest.approx(1.5)
    # Total actual = 26. Total error = 3 => WAPE = 3 / 26
    assert perf["WAPE"] == pytest.approx(3.0 / 26.0)
    # Predicted sum = 25, Actual sum = 26 => Bias = (25 - 26) / 26 = -1/26
    assert perf["Bias"] == pytest.approx(-1.0 / 26.0)
