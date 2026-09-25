"""FastAPI endpoint contract and operational behavior test suite."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from inference.app import app
import inference.app as app_module
from inference.feature_builder import FeatureBuilder
from inference.router import PredictionRouter
from inference.logger import PredictionLogger


class MockModel:
    def predict(self, X):
        return np.array([4.2])


@pytest.fixture
def client_with_mock_state(tmp_path):
    dates = pd.date_range("2016-01-01", periods=60)
    sales = pd.DataFrame({"item_id": "FOODS_1_001", "store_id": "CA_1", "date": dates, "sales": np.ones(60) * 5.0})
    calendar = pd.DataFrame({"date": dates, "event_name_1": None, "event_name_2": None, "wm_yr_wk": 100})
    prices = pd.DataFrame({"item_id": "FOODS_1_001", "store_id": "CA_1", "date": dates, "sell_price": 2.5})
    segments = pd.DataFrame({"item_id": ["FOODS_1_001"], "store_id": ["CA_1"], "is_normal_volume": [True]})

    features = ["lag_1", "rolling_mean_7", "day_of_week", "sell_price"]

    app_module.model_bundle = {
        "version": "v1_test",
        "model": MockModel(),
        "metadata": {"registered_at": "2026-01-01", "lineage_id": "test_sha_123"},
        "features": features,
        "segments": segments,
        "metrics": {"MAE": 0.5, "RMSE": 1.0, "WAPE": 0.1, "Bias": 0.0},
    }
    app_module.feature_builder = FeatureBuilder(sales, calendar, prices)
    app_module.prediction_router = PredictionRouter(MockModel(), segments, sales)
    app_module.prediction_logger = PredictionLogger(log_dir=str(tmp_path / "logs"))

    return TestClient(app)


def test_api_health(client_with_mock_state):
    response = client_with_mock_state.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "v1_test"


def test_api_predict(client_with_mock_state):
    req_body = {
        "item_id": "FOODS_1_001",
        "store_id": "CA_1",
        "date": "2016-03-01",
    }
    response = client_with_mock_state.post("/predict", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == "FOODS_1_001"
    assert data["store_id"] == "CA_1"
    assert data["prediction"] == pytest.approx(4.2)
    assert data["model_used"] == "lightgbm"
    assert data["model_version"] == "v1_test"
    assert data["lineage_id"] == "test_sha_123"
