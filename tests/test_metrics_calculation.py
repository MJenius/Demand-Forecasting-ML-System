import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate_forecasts, evaluate_stratified_slices


def test_evaluate_forecasts_basic():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    m = evaluate_forecasts(y_true, y_pred)

    assert m["MAE"] == pytest.approx((2 + 2 + 3) / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["WAPE"] == pytest.approx(7.0 / 60.0)
    assert m["Bias"] == pytest.approx((3 - 0) / 60.0)
    assert m["MAPE"] == pytest.approx(np.mean([20.0, 10.0, 10.0]))


def test_evaluate_forecasts_zeros_and_negatives():
    y_true = [0.0, 0.0, 0.0]
    y_pred = [-1.0, 0.0, 2.0]
    m = evaluate_forecasts(y_true, y_pred)

    # Negative forecast should be clipped to 0
    # Clipped y_pred: [0.0, 0.0, 2.0]
    assert m["MAE"] == pytest.approx(2.0 / 3.0)
    assert np.isnan(m["MAPE"])  # No positive actuals
    assert m["Bias"] > 0  # Overforecasting


def test_evaluate_stratified_slices():
    df = pd.DataFrame({
        "target": [1.0, 2.0, 10.0, 15.0],
        "prediction": [1.0, 1.5, 9.0, 16.0],
        "item_id": ["i1", "i2", "i3", "i4"],
        "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
        "horizon": [7, 7, 28, 28],
        "mean_sales": [0.4, 1.0, 5.0, 10.0],
    })

    slices = evaluate_stratified_slices(df)
    assert "by_horizon" in slices
    assert len(slices["by_horizon"]) == 2
    assert "by_store" in slices
    assert len(slices["by_store"]) == 2
    assert "by_volume_tier" in slices
    tiers = {row["volume_tier"] for row in slices["by_volume_tier"]}
    assert tiers == {"intermittent", "low", "normal"}
