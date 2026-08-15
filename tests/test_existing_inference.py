import numpy as np
import pandas as pd

from inference.feature_builder import FeatureBuilder
from inference.router import PredictionRouter


class ConstantModel:
    def predict(self, _values):
        return np.array([2.5])


def test_existing_feature_builder_and_router_paths():
    dates = pd.date_range("2020-01-01", periods=30)
    sales = pd.DataFrame({"item_id": "i", "store_id": "s", "date": dates, "sales": np.arange(30, dtype=float)})
    calendar = pd.DataFrame({"date": dates, "event_name_1": None, "event_name_2": None})
    prices = pd.DataFrame({"item_id": "i", "store_id": "s", "date": dates, "sell_price": 1.0})
    names = ["lag_1", "rolling_mean_7", "day_of_week", "sell_price"]
    built = FeatureBuilder(sales, calendar, prices).build_features("i", "s", "2020-01-31", names)
    assert built["lag_1"] == 29
    assert built["rolling_mean_7"] == 26
    normal = pd.DataFrame({"item_id": ["i"], "store_id": ["s"], "is_normal_volume": [True]})
    prediction, used = PredictionRouter(ConstantModel(), normal, sales).predict("i", "s", "2020-01-31", built, names)
    assert (prediction, used) == (2.5, "lightgbm")


def test_existing_router_moving_average_fallback():
    dates = pd.date_range("2020-01-01", periods=7)
    sales = pd.DataFrame({"item_id": "i", "store_id": "s", "date": dates, "sales": np.arange(1, 8, dtype=float)})
    segments = pd.DataFrame(columns=["item_id", "store_id", "is_normal_volume"])
    prediction, used = PredictionRouter(ConstantModel(), segments, sales).predict("i", "s", "2020-01-08", {}, [])
    assert prediction == 4
    assert used == "moving_average"
