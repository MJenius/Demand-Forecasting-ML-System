"""Reproducible, leakage-safe training, evaluation, drift, and promotion pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import tomllib
import tracemalloc
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import psutil
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from src.evaluation.metrics import evaluate_forecasts
from src.registry.promotion_gate import evaluate_promotion
from src.registry.manager import ModelRegistry

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("demand_pipeline")
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FEATURE_EXCLUDE = {"date", "item_id", "store_id", "target"}



def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def full_run_memory_preflight(config: dict) -> None:
    """Keep enough headroom for one bounded full-data batch."""
    if config["data"]["sample_items"] < 3049:
        return
    available_gb = psutil.virtual_memory().available / 1024**3
    required_gb = config["data"].get("minimum_available_memory_gb", 1.6)
    if available_gb < required_gb:
        raise MemoryError(
            f"Full 3,049-item run requires {required_gb} GB free for one bounded batch; "
            f"this host has {available_gb:.1f} GB. Close another memory-intensive process and retry."
        )


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unavailable"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def benchmark(name: str, results: dict):
    process = psutil.Process()
    cpu_start = process.cpu_times()
    rss_start = process.memory_info().rss
    owns_trace = not tracemalloc.is_tracing()
    if owns_trace:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
        if owns_trace:
            tracemalloc.stop()
        cpu_end = process.cpu_times()
        results[name] = {
            "seconds": elapsed,
            "cpu_seconds": (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system),
            "rss_delta_mb": (process.memory_info().rss - rss_start) / 1024**2,
            "python_peak_mb": peak / 1024**2,
        }
        LOG.info("%s finished in %.2fs", name, elapsed)


def ensure_raw_data(raw_dir: Path) -> None:
    required = {"sales_train_validation.csv", "calendar.csv", "sell_prices.csv"}
    if required.issubset({p.name for p in raw_dir.glob("*.csv")}):
        return
    archive = raw_dir.parent / "m5-forecasting-accuracy.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Missing raw CSV files and archive: {archive}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in required:
            source.extract(name, raw_dir)


def validate_data(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame, config: dict, schema_path: Path) -> dict:
    required = {
        "sales": {"id", "item_id", "store_id"},
        "calendar": {"date", "d", "wm_yr_wk"},
        "prices": {"store_id", "item_id", "wm_yr_wk", "sell_price"},
    }
    frames = {"sales": sales, "calendar": calendar, "prices": prices}
    errors: list[str] = []
    for name, expected in required.items():
        missing = expected - set(frames[name].columns)
        if missing:
            errors.append(f"{name}: missing columns {sorted(missing)}")
    day_cols = [column for column in sales if column.startswith("d_")]
    if not day_cols:
        errors.append("sales: no d_* target columns")
    parsed_dates = pd.to_datetime(calendar.get("date"), errors="coerce")
    if parsed_dates.isna().any():
        errors.append(f"calendar: {int(parsed_dates.isna().sum())} invalid dates")
    if calendar.duplicated("d").any():
        errors.append(f"calendar: {int(calendar.duplicated('d').sum())} duplicate day keys")
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        errors.append("prices: duplicate store/item/week keys")
    if prices["sell_price"].isna().any() or (prices["sell_price"] <= 0).any():
        errors.append("prices: null or non-positive sell_price")
    if sales[day_cols].isna().any().any() or (sales[day_cols] < 0).any().any():
        errors.append("sales: null or negative demand")
    last_sales_day = max(int(column[2:]) for column in day_cols)
    last_calendar_day = calendar["d"].str.removeprefix("d_").astype(int).max()
    freshness_gap = int(last_calendar_day - last_sales_day)
    if freshness_gap > config["data"]["max_calendar_gap_days"]:
        errors.append(f"sales freshness gap is {freshness_gap} days")
    schema = {name: {column: str(dtype) for column, dtype in frame.dtypes.items()} for name, frame in frames.items()}
    schema_changed = False
    if schema_path.exists():
        previous = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_changed = previous != schema
        if schema_changed:
            errors.append("input schema changed from the recorded contract")
    else:
        atomic_json(schema_path, schema)
    report = {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "schema_changed": schema_changed,
        "freshness_gap_days": freshness_gap,
        "rows": {name: len(frame) for name, frame in frames.items()},
    }
    if errors:
        raise ValueError("Data quality failed: " + "; ".join(errors))
    return report


def load_raw(config: dict, artifacts: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = ROOT / "data" / "raw" / "m5"
    ensure_raw_data(raw)
    sales_path, calendar_path, prices_path = raw / "sales_train_validation.csv", raw / "calendar.csv", raw / "sell_prices.csv"
    header = pd.read_csv(sales_path, nrows=0)
    rng = np.random.default_rng(config["data"]["seed"])
    identifiers = pd.read_csv(sales_path, usecols=ID_COLS)
    items = np.sort(identifiers["item_id"].unique())
    selected = set(rng.choice(items, min(config["data"]["sample_items"], len(items)), replace=False))
    sales = pd.read_csv(sales_path, usecols=list(header.columns))
    sales = sales[sales["item_id"].isin(selected)].reset_index(drop=True)
    calendar = pd.read_csv(calendar_path)
    price_parts = []
    for part in pd.read_csv(prices_path, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"], chunksize=250_000):
        price_parts.append(part[part["item_id"].isin(selected)])
    prices = pd.concat(price_parts, ignore_index=True)
    del price_parts
    quality = validate_data(sales, calendar, prices, config, artifacts / "schema.json")
    dataset = {
        "source": "M5 Forecasting Accuracy",
        "archive_sha256": file_sha256(raw.parent / "m5-forecasting-accuracy.zip"),
        "sales_sha256": file_sha256(sales_path),
        "sample_seed": config["data"]["seed"],
        "sample_items": len(selected),
        "sales_rows_wide": len(sales),
    }
    return sales, calendar, prices, {"quality": quality, "dataset": dataset}


def build_features(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    day_cols = [column for column in sales if column.startswith("d_")]
    df = sales[["item_id", "store_id", *day_cols]].melt(
        id_vars=["item_id", "store_id"], var_name="d", value_name="target"
    )
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["is_event"] = calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).astype("int8")
    calendar["snap"] = calendar[["snap_CA", "snap_TX", "snap_WI"]].max(axis=1).astype("int8")
    df = df.merge(calendar[["d", "date", "wm_yr_wk", "is_event", "snap"]], on="d", validate="many_to_one")
    df = df.merge(prices[["item_id", "store_id", "wm_yr_wk", "sell_price"]], on=["item_id", "store_id", "wm_yr_wk"], how="left", validate="many_to_one")
    df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    groups = df.groupby(["item_id", "store_id"], sort=False)["target"]
    for lag in [horizon, horizon + 6, horizon + 13, horizon + 27, horizon + 55]:
        df[f"lag_{lag}"] = groups.shift(lag)
    shifted = groups.shift(horizon)
    shifted_groups = shifted.groupby([df["item_id"], df["store_id"]], sort=False)
    for window in [7, 14, 28, 56]:
        df[f"rolling_mean_{window}"] = shifted_groups.transform(lambda values: values.rolling(window, min_periods=window).mean())
        df[f"rolling_std_{window}"] = shifted_groups.transform(lambda values: values.rolling(window, min_periods=window).std())
    df["expanding_mean"] = shifted_groups.transform(lambda values: values.expanding(28).mean())
    historical = df[["date", "item_id", "store_id", "target"]].copy()
    for key, name in [("item_id", "item_mean"), ("store_id", "store_mean")]:
        aggregate = historical.groupby(["date", key], as_index=False)["target"].mean().sort_values([key, "date"])
        aggregate[name] = aggregate.groupby(key, sort=False)["target"].shift(horizon)
        df = df.merge(aggregate[["date", key, name]], on=["date", key], how="left", validate="many_to_one")
    global_mean = historical.groupby("date")["target"].mean().shift(horizon).rename("global_mean")
    df = df.merge(global_mean, left_on="date", right_index=True, how="left", validate="many_to_one")
    day = df["date"].dt.dayofyear
    df["dow"] = df["date"].dt.dayofweek.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["year_sin"] = np.sin(2 * np.pi * day / 365.25)
    df["year_cos"] = np.cos(2 * np.pi * day / 365.25)
    df["price_change_4w"] = df.groupby(["item_id", "store_id"], sort=False)["sell_price"].pct_change(28, fill_method=None)
    return df.drop(columns=["d", "wm_yr_wk"])


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in FEATURE_EXCLUDE]


def metrics(y_true, y_pred) -> dict[str, float]:
    return evaluate_forecasts(y_true, y_pred)


def psi(reference, current, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference, current = reference[np.isfinite(reference)], current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference[0], current) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, edges)[0] / len(reference)
    cur_counts = np.histogram(current, edges)[0] / len(current)
    ref_counts, cur_counts = np.clip(ref_counts, 1e-6, None), np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def drift_report(train: pd.DataFrame, model, features: list[str], config: dict) -> dict:
    bins = config["drift"]["bins"]
    end = train["date"].max()
    reference = train[(train["date"] > end - pd.Timedelta(days=56)) & (train["date"] <= end - pd.Timedelta(days=28))]
    current = train[train["date"] > end - pd.Timedelta(days=28)]
    # Calendar position changes by design; monitor observed/model-input distributions instead.
    monitored = [column for column in features if column not in {"dow", "month", "year_sin", "year_cos", "is_event", "snap"}]
    feature_psi = {column: psi(reference[column], current[column], bins) for column in monitored}
    target_psi = psi(reference["target"], current["target"], bins)
    prediction_psi = psi(model.predict(reference[features]), model.predict(current[features]), bins)
    values = [*feature_psi.values(), target_psi, prediction_psi]
    maximum = max((value for value in values if not np.isnan(value)), default=0.0)
    status = "DRIFT" if maximum >= config["drift"]["critical_psi"] else "WARNING" if maximum >= config["drift"]["warning_psi"] else "OK"
    return {"status": status, "max_psi": maximum, "feature_psi": feature_psi, "target_psi": target_psi, "prediction_psi": prediction_psi}


def fit_models(train: pd.DataFrame, features: list[str], config: dict):
    X, y = train[features], train["target"]
    simple = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0)).fit(X, y)
    lightgbm = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=config["models"]["lightgbm_estimators"],
        learning_rate=config["models"]["lightgbm_learning_rate"], num_leaves=31,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.8, random_state=config["data"]["seed"], n_jobs=-1, verbosity=-1,
    ).fit(X, y, categorical_feature=[features.index("dow"), features.index("month")])
    return simple, lightgbm


def baseline_predictions(train: pd.DataFrame, validation: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    indexed = train.set_index(["item_id", "store_id", "date"])["target"]
    output = {"seasonal_naive": [], "moving_average": []}
    for row in validation.itertuples():
        history = indexed.loc[(row.item_id, row.store_id)]
        available = history[history.index <= row.date - pd.Timedelta(days=horizon)]
        seasonal_date = row.date - pd.Timedelta(days=7 * math.ceil(horizon / 7))
        output["seasonal_naive"].append(float(history.get(seasonal_date, available.iloc[-1])))
        output["moving_average"].append(float(available.tail(28).mean()))
    return {name: np.asarray(values) for name, values in output.items()}


def evaluate_fold(df: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, config: dict, timings: dict) -> tuple[list[dict], dict, object, pd.DataFrame]:
    target_date = cutoff + pd.Timedelta(days=horizon)
    train = df[df["date"] <= cutoff].dropna(subset=feature_columns(df))
    validation = df[df["date"] == target_date].dropna(subset=feature_columns(df))
    if train["date"].nunique() < config["validation"]["min_train_days"] or validation.empty:
        raise ValueError(f"Insufficient data for cutoff={cutoff.date()} horizon={horizon}")
    features = feature_columns(df)
    with benchmark(f"train_h{horizon}_{cutoff.date()}", timings):
        simple, model = fit_models(train, features, config)

    # Compute volume threshold using only train historical data up to cutoff (zero leakage)
    vol_threshold = config["models"].get("volume_threshold", 1.5)
    sku_means = train.groupby(["item_id", "store_id"], as_index=False)["target"].mean()
    sku_means.rename(columns={"target": "mean_sales"}, inplace=True)
    sku_means["is_normal_volume"] = sku_means["mean_sales"] >= vol_threshold

    with benchmark(f"inference_h{horizon}_{cutoff.date()}", timings):
        baselines = baseline_predictions(train, validation, horizon)
        lgb_preds = np.clip(model.predict(validation[features]), 0, None)
        ridge_preds = np.clip(simple.predict(validation[features]), 0, None)

        # Build segmented hybrid forecast
        merged_val = validation[["item_id", "store_id"]].merge(sku_means, on=["item_id", "store_id"], how="left")
        is_normal = merged_val["is_normal_volume"].fillna(False).to_numpy()
        hybrid_preds = np.where(is_normal, lgb_preds, baselines["moving_average"])

        predictions = {
            **baselines,
            "ridge": ridge_preds,
            "lightgbm": lgb_preds,
            "hybrid_segmented": hybrid_preds,
        }
    rows = []
    for name, predicted in predictions.items():
        m = metrics(validation["target"], predicted)
        rows.append({"fold_cutoff": str(cutoff.date()), "target_date": str(target_date.date()), "horizon": horizon, "model": name, **m})
    drift = drift_report(train, model, features, config)
    return rows, drift, model, sku_means


def promotion_decision(current: dict, candidate: dict, config: dict | None = None) -> dict:
    return evaluate_promotion(current, candidate, config)



def track_run(report: dict, artifacts: Path, config: dict, config_path: Path | None = None) -> None:
    mlflow.set_tracking_uri((ROOT / config["tracking"]["uri"]).resolve().as_uri())
    mlflow.set_experiment(config["tracking"]["experiment"])
    with mlflow.start_run(run_name=f"pipeline-{datetime.now():%Y%m%d-%H%M%S}"):
        mlflow.log_params({
            "git_commit": report["git_commit"],
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()),
            "dataset_sha256": report["dataset"]["archive_sha256"],
            "sample_items": config["data"]["sample_items"],
            "sample_seed": config["data"]["seed"],
            "horizons": str(config["validation"]["horizons"]),
            "validation_folds": config["validation"]["n_folds"],
            "fold_spacing_days": config["validation"]["fold_spacing_days"],
            "lightgbm_estimators": config["models"]["lightgbm_estimators"],
            "lightgbm_learning_rate": config["models"]["lightgbm_learning_rate"],
            "psi_warning": config["drift"]["warning_psi"],
            "psi_critical": config["drift"]["critical_psi"],
            "features": json.dumps({path.stem: json.loads(path.read_text()) for path in (artifacts / "model").glob("features_*.json")}),
        })
        for row in report["comparison"]:
            for metric in ["MAE", "RMSE", "WAPE", "Bias", "MAPE"]:
                if metric in row and not np.isnan(row[metric]):
                    mlflow.log_metric(f"{row['model']}_h{row['horizon']}_{metric.lower()}", row[metric])

        for name in ["pipeline_report.json", "model_comparison.csv", "drift_report.json", "retraining_decision.json", "performance.json", "schema.json"]:
            path = artifacts / name
            if path.exists():
                mlflow.log_artifact(path, artifact_path="pipeline")
        mlflow.log_artifacts(artifacts / "model", artifact_path="pipeline/model")
        for path in [config_path or ROOT / "pipeline.toml", ROOT / "production_pipeline.py", ROOT / "streaming_pipeline.py", ROOT / "requirements.txt", ROOT / "pyproject.toml"]:
            if path.exists():
                mlflow.log_artifact(path, artifact_path="source")


def run(config_path: Path, resume: bool = False) -> dict:
    config = load_config(config_path)
    full_run_memory_preflight(config)
    if config["data"]["sample_items"] >= 3049:
        from streaming_pipeline import run_full

        return run_full(config_path, resume)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    setup_logging(artifacts / "pipeline.log")
    checkpoint_path = artifacts / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if resume and checkpoint_path.exists() else {"completed": []}
    timings: dict = {}
    started = time.perf_counter()
    try:
        if resume and "evaluation" in checkpoint["completed"] and (artifacts / "pipeline_report.json").exists():
            report = json.loads((artifacts / "pipeline_report.json").read_text(encoding="utf-8"))
            track_run(report, artifacts, config, config_path)
            checkpoint["completed"].append("tracking")
            checkpoint["status"] = "SUCCESS"
            checkpoint.pop("error", None)
            checkpoint.pop("failed_at", None)
            atomic_json(checkpoint_path, checkpoint)
            LOG.info("Pipeline recovered from evaluation checkpoint")
            return report
        with benchmark("load_and_validate", timings):
            sales, calendar, prices, source = load_raw(config, artifacts)
        checkpoint["completed"].append("data_quality")
        atomic_json(checkpoint_path, checkpoint)
        all_rows, drifts, final_models, model_history, latest_holdouts = [], [], {}, {}, {}
        max_date = pd.to_datetime(calendar.loc[calendar["d"].isin([c for c in sales if c.startswith("d_")]), "date"]).max()
        with benchmark("feature_engineering_and_validation", timings):
            latest_sku_segments = None
            for horizon in config["validation"]["horizons"]:
                featured = build_features(sales, calendar, prices, horizon)
                for fold in reversed(range(config["validation"]["n_folds"])):
                    cutoff = max_date - pd.Timedelta(days=horizon + fold * config["validation"]["fold_spacing_days"])
                    rows, drift, model, sku_means = evaluate_fold(featured, cutoff, horizon, config, timings)
                    all_rows.extend(rows)
                    drifts.append({"cutoff": str(cutoff.date()), "horizon": horizon, **drift})
                    final_models[horizon] = (model, feature_columns(featured))
                    model_history.setdefault(horizon, []).append((cutoff, model))
                    if fold == 0:
                        latest_sku_segments = sku_means.copy()
                        features = feature_columns(featured)
                        holdout = featured[featured["date"] == cutoff + pd.Timedelta(days=horizon)].dropna(subset=features)
                        latest_holdouts[horizon] = (holdout[features].copy(), holdout["target"].copy())
                del featured
                gc.collect()
        comparison = pd.DataFrame(all_rows)
        summary = comparison.groupby(["model", "horizon"])[["MAE", "RMSE", "WAPE", "Bias", "MAPE"]].mean().reset_index()
        summary.to_csv(artifacts / "model_comparison.csv", index=False)
        atomic_json(artifacts / "drift_report.json", drifts)
        checkpoint["completed"].append("evaluation")
        atomic_json(checkpoint_path, checkpoint)
        latest_drift = [max((row for row in drifts if row["horizon"] == horizon), key=lambda row: row["cutoff"]) for horizon in config["validation"]["horizons"]]
        retraining = {"triggered": any(row["status"] == "DRIFT" for row in latest_drift), "decision": "NOT_TRIGGERED"}
        if retraining["triggered"]:
            incumbent_rows, candidate_rows = [], []
            for horizon, history in model_history.items():
                history.sort(key=lambda item: item[0])
                latest_cutoff, candidate_model = history[-1]
                incumbent_model = history[-2][1]
                holdout_x, holdout_y = latest_holdouts[horizon]
                incumbent_rows.append(metrics(holdout_y, incumbent_model.predict(holdout_x)))
                candidate_rows.append(metrics(holdout_y, candidate_model.predict(holdout_x)))
            incumbent = pd.DataFrame(incumbent_rows).mean().to_dict()
            candidate = pd.DataFrame(candidate_rows).mean().to_dict()
            promotion_res = promotion_decision(incumbent, candidate, config["models"])
            retraining.update(promotion_res)
            retraining.update({"incumbent_metrics": incumbent, "candidate_metrics": candidate})
            if retraining["decision"] == "REJECT":
                final_models = {horizon: (history[-2][1], list(latest_holdouts[horizon][0].columns)) for horizon, history in model_history.items()}
        atomic_json(artifacts / "retraining_decision.json", retraining)

        # Register final primary model into versioned ModelRegistry
        registry = ModelRegistry()
        primary_horizon = config["validation"]["horizons"][0]
        primary_model, primary_features = final_models[primary_horizon]
        lgb_metrics_summary = summary[(summary["model"] == "lightgbm") & (summary["horizon"] == primary_horizon)]
        primary_metrics = lgb_metrics_summary[["MAE", "RMSE", "WAPE", "Bias", "MAPE"]].to_dict(orient="records")[0] if not lgb_metrics_summary.empty else {}

        reg_result = registry.register_candidate(
            model_object=primary_model,
            features=primary_features,
            metrics=primary_metrics,
            dataset_info=source["dataset"],
            validation_config=config["validation"],
            hyperparameters=config["models"],
            git_commit=git_commit(),
            git_dirty=bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()),
            sku_segments=latest_sku_segments,
            promotion_config=config["models"],
            model_type="lightgbm_hybrid",
        )
        LOG.info("Registered model in ModelRegistry: %s (Promoted: %s, Lineage: %s)", reg_result["registered_version"], reg_result["promoted"], reg_result["lineage_id"])

        model_dir = artifacts / "model"
        model_dir.mkdir(exist_ok=True)
        for horizon, (model, features) in final_models.items():
            joblib.dump(model, model_dir / f"lightgbm_h{horizon}.joblib")
            atomic_json(model_dir / f"features_h{horizon}.json", features)
        total_seconds = time.perf_counter() - started
        evaluated = len(sales) * config["validation"]["n_folds"] * len(config["validation"]["horizons"])
        timings["pipeline"] = {"seconds": total_seconds, "evaluated_forecasts": evaluated, "forecast_throughput_per_second": evaluated / total_seconds}
        atomic_json(artifacts / "performance.json", timings)
        report = {
            **source,
            "git_commit": git_commit(),
            "lineage_id": reg_result["lineage_id"],
            "registered_version": reg_result["registered_version"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "comparison": summary.to_dict(orient="records"),
            "drift": drifts,
            "retraining": retraining,
            "performance": timings,
        }

        atomic_json(artifacts / "pipeline_report.json", report)
        track_run(report, artifacts, config, config_path)
        checkpoint["completed"].append("tracking")
        checkpoint["status"] = "SUCCESS"
        checkpoint.pop("error", None)
        checkpoint.pop("failed_at", None)
        atomic_json(checkpoint_path, checkpoint)
        LOG.info("Pipeline completed successfully")
        return report
    except Exception as exc:
        checkpoint.update({"status": "FAILED", "error": str(exc), "failed_at": datetime.now(timezone.utc).isoformat()})
        atomic_json(checkpoint_path, checkpoint)
        LOG.exception("Pipeline failed; rerun with --resume after correcting the cause")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline.toml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
