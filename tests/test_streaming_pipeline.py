import numpy as np
import pandas as pd
import pytest

from streaming_pipeline import ParquetSequence, _iter_sales_batches, build_feature_cache, streaming_ridge, train_lightgbm


def test_sales_csv_is_streamed_in_fixed_float32_batches(tmp_path):
    path = tmp_path / "sales.csv"
    pd.DataFrame({
        "id": ["a", "b", "c"], "item_id": ["i1", "i2", "i3"], "store_id": ["s1", "s1", "s2"],
        "d_1": [1, 2, 3], "d_2": [4, 5, 6],
    }).to_csv(path, index=False)
    batches = list(_iter_sales_batches(path, ["d_1", "d_2"], 2))
    assert [len(batch[0]) for batch in batches] == [2, 1]
    assert batches[0][0].dtype == np.float32
    assert batches[1][0].tolist() == [[3, 6]]


def test_chunked_features_preserve_horizon_shift(tmp_path):
    days = 80
    wide = pd.DataFrame({
        "series_code": np.array([0, 1], np.int32),
        "item_code": np.array([0, 1], np.int16),
        "store_code": np.array([0, 0], np.int8),
        **{f"d_{day + 1}": np.array([day, day + 100], np.float32) for day in range(days)},
    })
    wide_path = tmp_path / "wide.parquet"
    wide.to_parquet(wide_path, index=False)
    aggregate_path = tmp_path / "aggregates.npz"
    np.savez(
        aggregate_path,
        item_mean=np.vstack([np.arange(days), np.arange(days) + 100]).astype(np.float32),
        store_mean=(np.arange(days) + 50).reshape(1, -1).astype(np.float32),
        global_mean=(np.arange(days) + 50).astype(np.float32),
        weeks=np.arange(12, dtype=np.int32),
    )
    price_path = tmp_path / "prices.npy"
    np.save(price_path, np.ones((2, 12), np.float32))
    calendar_path = tmp_path / "calendar.csv"
    dates = pd.date_range("2020-01-01", periods=days)
    pd.DataFrame({
        "date": dates, "wm_yr_wk": np.repeat(np.arange(12), 7)[:days],
        "event_name_1": None, "event_name_2": None, "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
    }).to_csv(calendar_path, index=False)
    manifest = {
        "aggregate_path": str(aggregate_path), "price_path": str(price_path), "calendar_path": str(calendar_path),
        "days": days, "wide_files": [str(wide_path)],
    }
    result = build_feature_cache(manifest, 7, {}, {})
    features = pd.read_parquet(result["files"][0])
    first = features.iloc[0]
    assert first["date_idx"] == 62
    assert first["target"] == 62
    assert first["lag_7"] == 55
    assert first["rolling_mean_7"] == np.mean(np.arange(49, 56))


def test_lazy_sequence_and_streaming_ridge(tmp_path):
    path = tmp_path / "features.parquet"
    x1 = np.arange(200, dtype=np.float32)
    x2 = np.ones(200, np.float32)
    pd.DataFrame({"date_idx": np.repeat(np.arange(40), 5), "x1": x1, "x2": x2, "target": 2 * x1 + 3}).to_parquet(path, index=False, row_group_size=40)
    sequence = ParquetSequence(str(path), 29, ["x1", "x2"])
    assert len(sequence) == 150
    assert sequence[7].tolist() == [7, 1]
    models = streaming_ridge([str(path)], [19, 29], ["x1", "x2"], {}, 1)
    prediction = models[-1].predict(pd.DataFrame({"x1": [160.0], "x2": [1.0]}))[0]
    assert abs(prediction - 323) < 0.1
    (tmp_path / "cache").mkdir()
    config = {"data": {"seed": 1}, "models": {"lightgbm_learning_rate": 0.1, "lightgbm_estimators": 5, "max_bin": 15}}
    booster = train_lightgbm([str(path), str(path)], 29, ["x1", "x2"], config, tmp_path, "test", {})
    assert booster.num_trees() == 5
    assert len(booster.predict(pd.DataFrame({"x1": [160.0], "x2": [1.0]}))) == 1
    config["models"]["lightgbm_estimators"] = 1
    with pytest.raises(ValueError, match="one boosting round"):
        train_lightgbm([str(path), str(path)], 29, ["x1", "x2"], config, tmp_path, "test", {})
