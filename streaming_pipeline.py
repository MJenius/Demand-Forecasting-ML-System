"""Memory-bounded full M5 pipeline used by production_pipeline.py."""

from __future__ import annotations

import gc
import csv
import json
import logging
import math
import os
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from production_pipeline import (
    ROOT,
    atomic_json,
    benchmark,
    ensure_raw_data,
    file_sha256,
    git_commit,
    load_config,
    metrics,
    promotion_decision,
    psi,
    setup_logging,
    track_run,
)

LOG = logging.getLogger("demand_pipeline")
CACHE_VERSION = 4
NON_DRIFT_FEATURES = {"dow", "month", "year_sin", "year_cos", "is_event", "snap"}


class PeakRSS:
    def __init__(self, report_path: Path | None = None):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.report_path = report_path
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        ticks = 0
        while not self._stop.wait(0.1):
            memory = self.process.memory_info()
            self.peak = max(self.peak, getattr(memory, "peak_wset", memory.rss))
            ticks += 1
            if self.report_path and ticks % 10 == 0:
                try:
                    atomic_json(self.report_path, {"peak_rss_mb": self.peak / 1024**2, "current_rss_mb": memory.rss / 1024**2})
                except OSError:
                    pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self.thread.join()
        memory = self.process.memory_info()
        self.peak = max(self.peak, getattr(memory, "peak_wset", memory.rss))


def feature_names(horizon: int) -> list[str]:
    return [
        "is_event", "snap", "sell_price",
        *[f"lag_{lag}" for lag in [horizon, horizon + 6, horizon + 13, horizon + 27, horizon + 55]],
        *[name for window in [7, 14, 28, 56] for name in (f"rolling_mean_{window}", f"rolling_std_{window}")],
        "expanding_mean", "item_mean", "store_mean", "global_mean",
        "dow", "month", "year_sin", "year_cos", "price_change_4w",
    ]


def _iter_sales_batches(path: Path, day_cols: list[str], batch_size: int):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        day_start = header.index(day_cols[0])
        if header[day_start:] != day_cols:
            raise ValueError("Data quality failed: sales day columns are missing or reordered")
        item_position, store_position = header.index("item_id"), header.index("store_id")
        values = np.empty((batch_size, len(day_cols)), np.float32)
        items, stores, used = [], [], 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Data quality failed: malformed sales row")
            values[used] = np.fromiter((np.nan if value == "" else float(value) for value in row[day_start:]), np.float32, count=len(day_cols))
            items.append(row[item_position])
            stores.append(row[store_position])
            used += 1
            if used == batch_size:
                yield values, items, stores
                values = np.empty_like(values)
                items, stores, used = [], [], 0
        if used:
            yield values[:used], items, stores


def prepare_source(config: dict, artifacts: Path, timings: dict) -> dict:
    raw = ROOT / "data" / "raw" / "m5"
    ensure_raw_data(raw)
    sales_path = raw / "sales_train_validation.csv"
    calendar_path = raw / "calendar.csv"
    prices_path = raw / "sell_prices.csv"
    sales_hash = file_sha256(sales_path)
    cache = artifacts / "cache" / f"v{CACHE_VERSION}-{sales_hash[:12]}"
    manifest_path = cache / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(Path(path).exists() for path in manifest["wide_files"]):
            LOG.info("Reusing validated wide-data cache")
            return manifest

    cache.mkdir(parents=True, exist_ok=True)
    wide_dir = cache / "wide"
    wide_dir.mkdir(exist_ok=True)
    ids = pd.read_csv(sales_path, usecols=["id", "item_id", "store_id"])
    if ids.duplicated(["item_id", "store_id"]).any() or ids["id"].duplicated().any():
        raise ValueError("Data quality failed: duplicate sales series")
    items = sorted(ids["item_id"].unique())
    stores = sorted(ids["store_id"].unique())
    if len(items) != 3049:
        raise ValueError(f"Expected 3,049 items, found {len(items):,}")
    item_lookup = {value: i for i, value in enumerate(items)}
    store_lookup = {value: i for i, value in enumerate(stores)}
    series_index = pd.MultiIndex.from_frame(ids[["item_id", "store_id"]])
    header = pd.read_csv(sales_path, nrows=0)
    day_cols = [column for column in header if column.startswith("d_")]
    n_days = len(day_cols)
    item_sum = np.zeros((len(items), n_days), np.float32)
    store_sum = np.zeros((len(stores), n_days), np.float32)
    item_count = np.zeros(len(items), np.int16)
    store_count = np.zeros(len(stores), np.int16)
    wide_files, negative_count, null_count = [], 0, 0
    batch_size = config["data"].get("series_batch_size", 120)
    with benchmark("source_scan_and_cache", timings):
        for part, (values, item_values, store_values) in enumerate(_iter_sales_batches(sales_path, day_cols, batch_size)):
            null_count += int(np.isnan(values).sum())
            negative_count += int((values < 0).sum())
            item_code = np.fromiter((item_lookup[value] for value in item_values), np.int32, count=len(item_values))
            store_code = np.fromiter((store_lookup[value] for value in store_values), np.int32, count=len(store_values))
            series_code = series_index.get_indexer(pd.MultiIndex.from_arrays([item_values, store_values])).astype(np.int32)
            for code in np.unique(item_code):
                rows = item_code == code
                item_sum[code] += values[rows].sum(axis=0, dtype=np.float32)
                item_count[code] += rows.sum()
            for code in np.unique(store_code):
                rows = store_code == code
                store_sum[code] += values[rows].sum(axis=0, dtype=np.float32)
                store_count[code] += rows.sum()
            output = pd.DataFrame(values, columns=day_cols)
            output.insert(0, "series_code", series_code)
            output.insert(1, "item_code", item_code.astype(np.int16))
            output.insert(2, "store_code", store_code.astype(np.int8))
            path = wide_dir / f"part-{part:03d}.parquet"
            temporary = path.with_suffix(".tmp.parquet")
            output.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
            wide_files.append(str(path.resolve()))
            del output, values, item_values, store_values
            gc.collect()
    if null_count or negative_count:
        raise ValueError(f"Data quality failed: {null_count} null and {negative_count} negative sales values")

    calendar = pd.read_csv(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    if calendar["date"].isna().any() or calendar.duplicated("d").any():
        raise ValueError("Data quality failed: invalid or duplicate calendar dates")
    calendar = calendar.iloc[:n_days].copy()
    weeks = np.sort(calendar["wm_yr_wk"].unique())
    week_lookup = {int(value): i for i, value in enumerate(weeks)}
    price_matrix = np.full((len(ids), len(weeks)), np.nan, np.float32)
    duplicate_prices = invalid_prices = 0
    with benchmark("price_validation_and_index", timings):
        for chunk in pd.read_csv(prices_path, usecols=["item_id", "store_id", "wm_yr_wk", "sell_price"], chunksize=100_000):
            invalid_prices += int(chunk["sell_price"].isna().sum() + (chunk["sell_price"] <= 0).sum())
            duplicate_prices += int(chunk.duplicated(["item_id", "store_id", "wm_yr_wk"]).sum())
            series_code = series_index.get_indexer(pd.MultiIndex.from_frame(chunk[["item_id", "store_id"]]))
            week_code = chunk["wm_yr_wk"].map(week_lookup).fillna(-1).to_numpy(np.int32)
            valid = (series_code >= 0) & (week_code >= 0)
            existing = price_matrix[series_code[valid], week_code[valid]]
            duplicate_prices += int(np.isfinite(existing).sum())
            price_matrix[series_code[valid], week_code[valid]] = chunk.loc[valid, "sell_price"].to_numpy(np.float32)
            del chunk
    if duplicate_prices or invalid_prices:
        raise ValueError(f"Data quality failed: {duplicate_prices} duplicate and {invalid_prices} invalid price rows")
    price_path = cache / "prices.npy"
    np.save(price_path, price_matrix)
    aggregate_path = cache / "aggregates.npz"
    np.savez(
        aggregate_path,
        item_mean=(item_sum / item_count[:, None]).astype(np.float32),
        store_mean=(store_sum / store_count[:, None]).astype(np.float32),
        global_mean=(item_sum.sum(axis=0) / len(ids)).astype(np.float32),
        weeks=weeks.astype(np.int32),
    )
    all_calendar_days = pd.read_csv(calendar_path, usecols=["d"])["d"].str.removeprefix("d_").astype(int)
    freshness_gap = int(all_calendar_days.max() - n_days)
    if freshness_gap > config["data"]["max_calendar_gap_days"]:
        raise ValueError(f"Data quality failed: sales freshness gap is {freshness_gap} days")
    schema = {
        "sales": {column: str(dtype) for column, dtype in pd.read_csv(sales_path, nrows=10).dtypes.items()},
        "calendar": {column: str(dtype) for column, dtype in pd.read_csv(calendar_path).dtypes.items()},
        "prices": {column: str(dtype) for column, dtype in pd.read_csv(prices_path, nrows=10).dtypes.items()},
    }
    schema_path = artifacts / "schema.json"
    if schema_path.exists() and json.loads(schema_path.read_text(encoding="utf-8")) != schema:
        raise ValueError("Data quality failed: input schema changed from the recorded contract")
    atomic_json(schema_path, schema)
    manifest = {
        "cache_version": CACHE_VERSION,
        "sales_sha256": sales_hash,
        "archive_sha256": file_sha256(raw.parent / "m5-forecasting-accuracy.zip"),
        "wide_files": wide_files,
        "price_path": str(price_path.resolve()),
        "aggregate_path": str(aggregate_path.resolve()),
        "calendar_path": str(calendar_path.resolve()),
        "items": len(items), "stores": len(stores), "series": len(ids), "days": n_days,
        "historical_records": len(ids) * n_days,
        "freshness_gap_days": freshness_gap,
        "quality": {"status": "PASS", "null_sales": 0, "negative_sales": 0, "duplicate_sales": 0, "duplicate_prices": 0, "invalid_prices": 0},
    }
    atomic_json(manifest_path, manifest)
    del item_sum, store_sum, price_matrix, ids
    gc.collect()
    return manifest


def _rolling(y: np.ndarray, target_days: np.ndarray, horizon: int, window: int, squared=False) -> np.ndarray:
    source = np.square(y, dtype=np.float64) if squared else y
    cumulative = np.pad(np.cumsum(source, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
    end = target_days - horizon + 1
    return cumulative[:, end] - cumulative[:, end - window]


def build_feature_cache(manifest: dict, horizon: int, config: dict, timings: dict) -> dict:
    cache = Path(manifest["aggregate_path"]).parent
    feature_dir = cache / f"features-v{CACHE_VERSION}-h{horizon}"
    feature_dir.mkdir(exist_ok=True)
    feature_manifest_path = feature_dir / "manifest.json"
    if feature_manifest_path.exists():
        saved = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if all(Path(path).exists() for path in saved["files"]):
            LOG.info("Reusing horizon-%s feature cache", horizon)
            return saved
    aggregates = np.load(manifest["aggregate_path"], mmap_mode="r")
    prices = np.load(manifest["price_path"], mmap_mode="r")
    calendar = pd.read_csv(manifest["calendar_path"]).iloc[:manifest["days"]].copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    weeks = aggregates["weeks"]
    week_lookup = {int(value): i for i, value in enumerate(weeks)}
    week_code = calendar["wm_yr_wk"].map(week_lookup).to_numpy(np.int16)
    is_event = calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).to_numpy(np.int8)
    snap = calendar[["snap_CA", "snap_TX", "snap_WI"]].max(axis=1).to_numpy(np.int8)
    dow = calendar["date"].dt.dayofweek.to_numpy(np.int8)
    month = calendar["date"].dt.month.to_numpy(np.int8)
    day_of_year = calendar["date"].dt.dayofyear.to_numpy(np.float32)
    year_sin = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    year_cos = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    start = horizon + 55
    target_days = np.arange(start, manifest["days"], dtype=np.int32)
    names = feature_names(horizon)
    output_files, rows = [], 0
    with benchmark(f"feature_generation_h{horizon}", timings):
        for part, wide_path in enumerate(manifest["wide_files"]):
            output_path = feature_dir / f"part-{part:03d}.parquet"
            if output_path.exists():
                output_files.append(str(output_path.resolve()))
                rows += pq.ParquetFile(output_path).metadata.num_rows
                continue
            wide = pd.read_parquet(wide_path)
            series_code = wide.pop("series_code").to_numpy(np.int32)
            item_code = wide.pop("item_code").to_numpy(np.int32)
            store_code = wide.pop("store_code").to_numpy(np.int32)
            y = wide.to_numpy(np.float32, copy=False)
            n_series, n_dates = len(y), len(target_days)
            flatten = lambda values: np.asarray(values, dtype=np.float32).T.reshape(-1)
            current_price = prices[series_code[:, None], week_code[target_days][None, :]]
            previous_price = prices[series_code[:, None], week_code[target_days - 28][None, :]]
            price_change = np.divide(current_price - previous_price, previous_price, out=np.full_like(current_price, np.nan), where=previous_price > 0)
            columns: dict[str, np.ndarray] = {
                "date_idx": np.repeat(target_days.astype(np.int16), n_series),
                "series_code": np.tile(series_code, n_dates),
                "target": flatten(y[:, target_days]),
                "is_event": np.repeat(is_event[target_days], n_series),
                "snap": np.repeat(snap[target_days], n_series),
                "sell_price": flatten(current_price),
            }
            for lag in [horizon, horizon + 6, horizon + 13, horizon + 27, horizon + 55]:
                columns[f"lag_{lag}"] = flatten(y[:, target_days - lag])
            for window in [7, 14, 28, 56]:
                sums = _rolling(y, target_days, horizon, window)
                sums_squared = _rolling(y, target_days, horizon, window, squared=True)
                columns[f"rolling_mean_{window}"] = flatten(sums / window)
                variance = np.maximum((sums_squared - sums * sums / window) / (window - 1), 0)
                columns[f"rolling_std_{window}"] = flatten(np.sqrt(variance))
                del sums, sums_squared, variance
            history_end = target_days - horizon + 1
            cumulative = np.cumsum(y, axis=1, dtype=np.float64)
            columns["expanding_mean"] = flatten(cumulative[:, history_end - 1] / history_end)
            columns["item_mean"] = flatten(aggregates["item_mean"][item_code[:, None], target_days - horizon])
            columns["store_mean"] = flatten(aggregates["store_mean"][store_code[:, None], target_days - horizon])
            columns["global_mean"] = np.repeat(aggregates["global_mean"][target_days - horizon], n_series)
            columns["dow"] = np.repeat(dow[target_days], n_series)
            columns["month"] = np.repeat(month[target_days], n_series)
            columns["year_sin"] = np.repeat(year_sin[target_days], n_series)
            columns["year_cos"] = np.repeat(year_cos[target_days], n_series)
            columns["price_change_4w"] = flatten(price_change)
            valid = np.isfinite(columns["sell_price"]) & np.isfinite(columns["price_change_4w"])
            table = pa.table({name: pa.array(values[valid]) for name, values in columns.items()})
            temporary = output_path.with_suffix(".tmp.parquet")
            pq.write_table(table, temporary, compression="zstd", compression_level=1, row_group_size=65_536)
            temporary.replace(output_path)
            count = len(table)
            rows += count
            output_files.append(str(output_path.resolve()))
            LOG.info("h%s feature part %s/%s: %s rows", horizon, part + 1, len(manifest["wide_files"]), f"{count:,}")
            del wide, y, columns, table, current_price, previous_price, price_change, cumulative, valid
            gc.collect()
    result = {
        "horizon": horizon, "start_day_idx": start, "files": output_files, "rows": rows,
        "features": names, "first_date": str(calendar.iloc[start]["date"].date()),
        "last_date": str(calendar.iloc[manifest["days"] - 1]["date"].date()),
    }
    atomic_json(feature_manifest_path, result)
    return result


@lru_cache(maxsize=128)
def _parquet(path: str) -> pq.ParquetFile:
    return pq.ParquetFile(path)


@lru_cache(maxsize=4)
def _row_group_matrix(path: str, group: int, columns: tuple[str, ...]) -> np.ndarray:
    table = _parquet(path).read_row_group(group, columns=list(columns))
    return np.column_stack([table[column].to_numpy(zero_copy_only=False) for column in columns]).astype(np.float32, copy=False)


def _row_group_slice(path: str, group: int, start: int, stop: int, columns: list[str]) -> np.ndarray:
    parquet = _parquet(path)
    arrays = [
        parquet.read_row_group(group, columns=[column])[column].slice(start, stop - start).to_numpy(zero_copy_only=False)
        for column in columns
    ]
    return np.column_stack(arrays).astype(np.float32, copy=False)


@lru_cache(maxsize=128)
def _row_group_offsets(path: str) -> np.ndarray:
    parquet = _parquet(path)
    return np.cumsum([0, *[parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]])


def read_rows(path: str, start: int, stop: int, columns: list[str]) -> np.ndarray:
    if stop <= start:
        return np.empty((0, len(columns)), np.float32)
    offsets = _row_group_offsets(path)
    first = int(np.searchsorted(offsets, start, side="right") - 1)
    last = int(np.searchsorted(offsets, stop - 1, side="right") - 1)
    pieces = []
    for group in range(first, last + 1):
        local_start = max(start, offsets[group]) - offsets[group]
        local_stop = min(stop, offsets[group + 1]) - offsets[group]
        if local_start == 0 and local_stop == offsets[group + 1] - offsets[group]:
            pieces.append(_row_group_matrix(path, group, tuple(columns)))
        else:
            pieces.append(_row_group_slice(path, group, int(local_start), int(local_stop), columns))
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


@lru_cache(maxsize=1024)
def row_limit(path: str, cutoff: int) -> int:
    dates = pq.read_table(path, columns=["date_idx"])["date_idx"].to_numpy(zero_copy_only=False)
    return int(np.searchsorted(dates, cutoff, side="right"))


class ParquetSequence(lgb.Sequence):
    batch_size = 65_536

    def __init__(self, path: str, cutoff: int, features: list[str]):
        self.path, self.features = path, features
        self.length = row_limit(path, cutoff)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            if index < 0 or index >= self.length:
                raise IndexError(index)
            return read_rows(self.path, int(index), int(index) + 1, self.features)[0].astype(np.float64)
        if isinstance(index, slice):
            return read_rows(self.path, index.start or 0, min(index.stop, self.length), self.features).astype(np.float64)
        if isinstance(index, list):
            return np.stack([self[item] for item in index])
        raise TypeError(type(index).__name__)


class RidgeModel:
    def __init__(self, coefficient: np.ndarray, intercept: float):
        self.coefficient, self.intercept = coefficient, intercept

    def predict(self, frame):
        values = frame.to_numpy(np.float32, copy=False) if isinstance(frame, pd.DataFrame) else np.asarray(frame)
        return values @ self.coefficient + self.intercept


def streaming_ridge(files: list[str], cutoffs: list[int], features: list[str], timings: dict, horizon: int) -> list[RidgeModel]:
    width = len(features)
    increments = [dict(xtx=np.zeros((width, width)), xty=np.zeros(width), sx=np.zeros(width), sy=0.0, n=0) for _ in cutoffs]
    with benchmark(f"ridge_training_h{horizon}", timings):
        for path in files:
            boundaries = [row_limit(path, cutoff) for cutoff in cutoffs]
            start = 0
            for slot, stop in enumerate(boundaries):
                for offset in range(start, stop, 65_536):
                    batch = read_rows(path, offset, min(offset + 65_536, stop), [*features, "target"]).astype(np.float64)
                    x, y = batch[:, :-1], batch[:, -1]
                    stats = increments[slot]
                    stats["xtx"] += x.T @ x
                    stats["xty"] += x.T @ y
                    stats["sx"] += x.sum(axis=0)
                    stats["sy"] += y.sum()
                    stats["n"] += len(y)
                start = stop
    models = []
    total = dict(xtx=np.zeros((width, width)), xty=np.zeros(width), sx=np.zeros(width), sy=0.0, n=0)
    for stats in increments:
        for key in total:
            total[key] += stats[key]
        mean_x, mean_y = total["sx"] / total["n"], total["sy"] / total["n"]
        centered_xx = total["xtx"] - total["n"] * np.outer(mean_x, mean_x)
        centered_xy = total["xty"] - total["n"] * mean_x * mean_y
        coefficient = np.linalg.solve(centered_xx + np.eye(width), centered_xy)
        models.append(RidgeModel(coefficient.astype(np.float32), float(mean_y - mean_x @ coefficient)))
    return models


def train_lightgbm(files: list[str], cutoff: int, features: list[str], config: dict, artifacts: Path, tag: str, timings: dict):
    params = {
        "objective": "regression_l1", "metric": "l1", "learning_rate": config["models"]["lightgbm_learning_rate"],
        "num_leaves": 31, "min_data_in_leaf": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "seed": config["data"]["seed"], "verbosity": -1, "num_threads": min(os.cpu_count() or 1, 2),
        "max_bin": config["models"].get("max_bin", 31), "force_col_wise": True, "histogram_pool_size": 32,
    }
    with benchmark(f"lightgbm_training_{tag}", timings):
        order = list(files)
        np.random.default_rng(config["data"]["seed"] + cutoff).shuffle(order)
        total_rounds = config["models"]["lightgbm_estimators"]
        if total_rounds < len(order):
            raise ValueError("LightGBM needs at least one boosting round per full-data batch")
        rounds_per_file, remainder = divmod(total_rounds, len(order))
        model = None
        categorical = [feature for feature in ["dow", "month"] if feature in features]
        for batch_number, path in enumerate(order):
            rounds = rounds_per_file + (batch_number < remainder)
            if not rounds:
                continue
            stop = row_limit(path, cutoff)
            predictors = np.concatenate([
                read_rows(path, start, min(start + ParquetSequence.batch_size, stop), features)
                for start in range(0, stop, ParquetSequence.batch_size)
            ])
            labels = np.concatenate([
                read_rows(path, start, min(start + ParquetSequence.batch_size, stop), ["target"])[:, 0]
                for start in range(0, stop, ParquetSequence.batch_size)
            ])
            dataset = lgb.Dataset(predictors, label=labels, feature_name=features, categorical_feature=categorical, params=params, free_raw_data=True)
            model = lgb.train(params, dataset, num_boost_round=rounds, init_model=model)
            model.free_dataset()
            del dataset, predictors, labels
            _row_group_matrix.cache_clear()
            gc.collect()
            if (batch_number + 1) % 10 == 0 or batch_number + 1 == len(order):
                LOG.info("%s LightGBM batch %s/%s", tag, batch_number + 1, len(order))
    gc.collect()
    return model


def load_date(files: list[str], day_idx: int, features: list[str]) -> pd.DataFrame:
    parts = []
    for path in files:
        start, stop = row_limit(path, day_idx - 1), row_limit(path, day_idx)
        if stop > start:
            matrix = read_rows(path, start, stop, [*features, "target"])
            parts.append(pd.DataFrame(matrix, columns=[*features, "target"]))
    return pd.concat(parts, ignore_index=True)


def load_window(files: list[str], start_day: int, end_day: int, features: list[str]) -> pd.DataFrame:
    parts = []
    for path in files:
        start, stop = row_limit(path, start_day - 1), row_limit(path, end_day)
        if stop > start:
            matrix = read_rows(path, start, stop, [*features, "target", "date_idx"])
            parts.append(pd.DataFrame(matrix, columns=[*features, "target", "date_idx"]))
    return pd.concat(parts, ignore_index=True)


def drift_for_fold(files: list[str], cutoff: int, model, features: list[str], config: dict) -> dict:
    reference = load_window(files, cutoff - 55, cutoff - 28, features)
    current = load_window(files, cutoff - 27, cutoff, features)
    monitored = [feature for feature in features if feature not in NON_DRIFT_FEATURES]
    feature_psi = {feature: psi(reference[feature], current[feature], config["drift"]["bins"]) for feature in monitored}
    target_psi = psi(reference["target"], current["target"], config["drift"]["bins"])
    prediction_psi = psi(model.predict(reference[features]), model.predict(current[features]), config["drift"]["bins"])
    values = [*feature_psi.values(), target_psi, prediction_psi]
    maximum = max((value for value in values if not np.isnan(value)), default=0.0)
    status = "DRIFT" if maximum >= config["drift"]["critical_psi"] else "WARNING" if maximum >= config["drift"]["warning_psi"] else "OK"
    del reference, current
    gc.collect()
    return {"status": status, "max_psi": maximum, "feature_psi": feature_psi, "target_psi": target_psi, "prediction_psi": prediction_psi}


def run_full(config_path: Path, resume=False) -> dict:
    config = load_config(config_path)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    setup_logging(artifacts / "pipeline.log")
    timings: dict = {}
    started = time.perf_counter()
    checkpoint_path = artifacts / "checkpoint.json"
    checkpoint = {"completed": [], "mode": "full-3049"}
    atomic_json(checkpoint_path, checkpoint)
    with PeakRSS(artifacts / "memory_live.json") as memory:
        try:
            manifest = prepare_source(config, artifacts, timings)
            checkpoint["completed"].append("data_quality")
            atomic_json(checkpoint_path, checkpoint)
            calendar = pd.read_csv(manifest["calendar_path"])
            calendar_dates = pd.to_datetime(calendar["date"]).iloc[:manifest["days"]].reset_index(drop=True)
            max_day = manifest["days"] - 1
            comparisons, drifts, model_pairs, validation_cache = [], [], {}, {}
            for horizon in config["validation"]["horizons"]:
                feature_manifest = build_feature_cache(manifest, horizon, config, timings)
                files, features = feature_manifest["files"], feature_manifest["features"]
                cutoffs = [max_day - horizon - fold * config["validation"]["fold_spacing_days"] for fold in reversed(range(config["validation"]["n_folds"]))]
                ridge_models = streaming_ridge(files, cutoffs, features, timings, horizon)
                recent_models = deque(maxlen=2)
                for fold, cutoff in enumerate(cutoffs):
                    tag = f"h{horizon}-{calendar_dates.iloc[cutoff].date()}"
                    model = train_lightgbm(files, cutoff, features, config, artifacts, tag, timings)
                    target_day = cutoff + horizon
                    validation = load_date(files, target_day, features)
                    predictions = {
                        "seasonal_naive": validation[f"lag_{7 * math.ceil(horizon / 7)}"].to_numpy(),
                        "moving_average": validation["rolling_mean_28"].to_numpy(),
                        "ridge": np.clip(ridge_models[fold].predict(validation[features]), 0, None),
                        "lightgbm": np.clip(model.predict(validation[features]), 0, None),
                    }
                    for name, predicted in predictions.items():
                        comparisons.append({
                            "fold_cutoff": str(calendar_dates.iloc[cutoff].date()),
                            "target_date": str(calendar_dates.iloc[target_day].date()),
                            "horizon": horizon, "model": name, **metrics(validation["target"], predicted),
                        })
                    drift = drift_for_fold(files, cutoff, model, features, config)
                    drifts.append({"cutoff": str(calendar_dates.iloc[cutoff].date()), "horizon": horizon, **drift})
                    recent_models.append((cutoff, model))
                    validation_cache[horizon] = (validation, features)
                    del predictions
                    gc.collect()
                model_pairs[horizon] = list(recent_models)
                del ridge_models
                _row_group_matrix.cache_clear()
                gc.collect()
            comparison = pd.DataFrame(comparisons)
            summary = comparison.groupby(["model", "horizon"])[["MAE", "RMSE", "MAPE"]].mean().reset_index()
            summary.to_csv(artifacts / "model_comparison.csv", index=False)
            atomic_json(artifacts / "drift_report.json", drifts)
            checkpoint["completed"].append("evaluation")
            atomic_json(checkpoint_path, checkpoint)
            latest_drift = [max((row for row in drifts if row["horizon"] == horizon), key=lambda row: row["cutoff"]) for horizon in config["validation"]["horizons"]]
            retraining = {"triggered": any(row["status"] == "DRIFT" for row in latest_drift), "decision": "NOT_TRIGGERED"}
            if retraining["triggered"]:
                incumbent_rows, candidate_rows = [], []
                for horizon, pairs in model_pairs.items():
                    validation, features = validation_cache[horizon]
                    incumbent, candidate = pairs[-2][1], pairs[-1][1]
                    incumbent_rows.append(metrics(validation["target"], incumbent.predict(validation[features])))
                    candidate_rows.append(metrics(validation["target"], candidate.predict(validation[features])))
                incumbent_metrics = pd.DataFrame(incumbent_rows).mean().to_dict()
                candidate_metrics = pd.DataFrame(candidate_rows).mean().to_dict()
                retraining.update(promotion_decision(incumbent_metrics, candidate_metrics, config["models"]["promotion_min_improvement"]))
                retraining.update({"incumbent_metrics": incumbent_metrics, "candidate_metrics": candidate_metrics})
                production_models = {horizon: pairs[-1 if retraining["decision"] == "PROMOTE" else -2][1] for horizon, pairs in model_pairs.items()}
            else:
                production_models = {horizon: pairs[-1][1] for horizon, pairs in model_pairs.items()}
            atomic_json(artifacts / "retraining_decision.json", retraining)
            for horizon, model in production_models.items():
                validation, features = validation_cache[horizon]
                matrix = validation[features]
                model.predict(matrix)
                started_inference = time.perf_counter()
                for _ in range(10):
                    prediction = model.predict(matrix)
                inference_seconds = (time.perf_counter() - started_inference) / 10
                timings[f"inference_h{horizon}"] = {
                    "seconds": inference_seconds, "rows": len(matrix),
                    "throughput_rows_per_second": len(matrix) / inference_seconds,
                }
                del matrix, prediction
            model_dir = artifacts / "model"
            model_dir.mkdir(exist_ok=True)
            for horizon, model in production_models.items():
                joblib.dump(model, model_dir / f"lightgbm_h{horizon}.joblib")
                atomic_json(model_dir / f"features_h{horizon}.json", validation_cache[horizon][1])
            total_seconds = time.perf_counter() - started
            evaluated = manifest["series"] * config["validation"]["n_folds"] * len(config["validation"]["horizons"])
            timings["pipeline"] = {
                "seconds": total_seconds, "evaluated_forecasts": evaluated,
                "forecast_throughput_per_second": evaluated / total_seconds,
                "peak_rss_mb": memory.peak / 1024**2,
            }
            atomic_json(artifacts / "performance.json", timings)
            report = {
                "quality": {**manifest["quality"], "freshness_gap_days": manifest["freshness_gap_days"]},
                "dataset": {
                    "source": "M5 Forecasting Accuracy", "archive_sha256": manifest["archive_sha256"],
                    "sales_sha256": manifest["sales_sha256"], "sample_seed": config["data"]["seed"],
                    "sample_items": manifest["items"], "stores": manifest["stores"], "series": manifest["series"],
                    "historical_records": manifest["historical_records"],
                    "historical_period": {"start": str(calendar_dates.iloc[0].date()), "end": str(calendar_dates.iloc[-1].date())},
                    "evaluation_periods": sorted(comparison["target_date"].unique().tolist()),
                },
                "git_commit": git_commit(), "comparison": summary.to_dict(orient="records"),
                "drift": drifts, "retraining": retraining, "performance": timings,
            }
            atomic_json(artifacts / "pipeline_report.json", report)
            track_run(report, artifacts, config, config_path)
            checkpoint.update({"completed": ["data_quality", "features", "evaluation", "tracking"], "status": "SUCCESS"})
            atomic_json(checkpoint_path, checkpoint)
            LOG.info("Full 3,049-item pipeline completed successfully")
            return report
        except Exception as exc:
            checkpoint.update({"status": "FAILED", "error": str(exc)})
            atomic_json(checkpoint_path, checkpoint)
            LOG.exception("Full pipeline failed")
            raise
