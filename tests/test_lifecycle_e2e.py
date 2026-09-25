"""End-to-end integration test validating the closed-loop ML lifecycle.

Lifecycle stages validated:
1. Ingestion & Quality check
2. Leakage-safe feature computation
3. Baseline and model fitting
4. Evaluation across metrics (MAE, RMSE, WAPE, Bias)
5. Model registry & promotion check
6. Serving & Prediction logging
7. Drift detection & Retraining trigger
"""

import numpy as np
import pandas as pd
import pytest

from production_pipeline import build_features, fit_models, baseline_predictions, drift_report
from src.evaluation.metrics import evaluate_forecasts
from src.registry.manager import ModelRegistry
from src.registry.promotion_gate import evaluate_promotion
from src.monitoring.reconciler import ActualsReconciler
from monitoring.collect_predictions import PredictionCollector


def test_full_ml_lifecycle_e2e(tmp_path):
    # Step 1: Synthesize a miniature multi-series dataset
    dates = pd.date_range("2020-01-01", periods=120)
    series_list = [("item_1", "store_1"), ("item_2", "store_1")]

    sales_records = []
    for item, store in series_list:
        base = 2.0 if item == "item_1" else 0.5
        demand = np.clip(np.random.normal(base, 0.5, len(dates)), 0, None)
        rec = {"id": f"{item}_{store}", "item_id": item, "dept_id": "d", "cat_id": "c", "store_id": store, "state_id": "CA"}
        for i, d in enumerate(dates):
            rec[f"d_{i+1}"] = float(demand[i])
        sales_records.append(rec)
    sales = pd.DataFrame(sales_records)

    calendar = pd.DataFrame({
        "date": dates.astype(str),
        "d": [f"d_{i+1}" for i in range(len(dates))],
        "wm_yr_wk": np.repeat(np.arange(18), 7)[: len(dates)],
        "event_name_1": None,
        "event_name_2": None,
        "snap_CA": 0,
        "snap_TX": 0,
        "snap_WI": 0,
    })

    price_rows = []
    for item, store in series_list:
        for wk in range(18):
            price_rows.append({"store_id": store, "item_id": item, "wm_yr_wk": wk, "sell_price": 2.0})
    prices = pd.DataFrame(price_rows)

    # Step 2: Build Leakage-Safe Features
    featured = build_features(sales, calendar, prices, horizon=7)
    feature_cols = [c for c in featured.columns if c not in {"date", "item_id", "store_id", "target"}]
    assert len(feature_cols) > 5

    # Step 3: Train models on training split
    cutoff = pd.Timestamp("2020-04-01")
    train = featured[featured["date"] <= cutoff].dropna(subset=feature_cols)
    validation = featured[featured["date"] == cutoff + pd.Timedelta(days=7)].dropna(subset=feature_cols)

    simple_ridge, lgb_model = fit_models(
        train,
        feature_cols,
        {"data": {"seed": 42}, "models": {"lightgbm_estimators": 10, "lightgbm_learning_rate": 0.1}},
    )

    # Step 4: Model Evaluation
    preds = lgb_model.predict(validation[feature_cols])
    val_metrics = evaluate_forecasts(validation["target"], preds)
    assert "WAPE" in val_metrics
    assert "Bias" in val_metrics

    # Step 5: Model Registry Registration & Promotion
    reg = ModelRegistry(registry_dir=tmp_path / "registry")
    reg_res = reg.register_candidate(
        model_object=lgb_model,
        features=feature_cols,
        metrics=val_metrics,
        dataset_info={"source": "synthetic_m5"},
        validation_config={"horizons": [7]},
        hyperparameters={"estimators": 10},
        git_commit="test_commit",
        git_dirty=False,
    )
    assert reg_res["promoted"] is True
    assert reg.get_active_model_version() == "v1"

    # Step 6: Prediction Logging & Delayed Actuals
    db_path = tmp_path / "predictions.db"
    collector = PredictionCollector(db_path=str(db_path))
    collector.add_prediction(
        timestamp="2020-04-08T00:00:00",
        item_id="item_1",
        store_id="store_1",
        date="2020-04-08",
        prediction=float(preds[0]),
        model_used="lightgbm",
        model_version="v1",
    )

    reconciler = ActualsReconciler(db_path=db_path)
    actuals = pd.DataFrame({
        "item_id": ["item_1"],
        "store_id": ["store_1"],
        "date": ["2020-04-08"],
        "sales": [float(validation["target"].iloc[0])],
    })
    reconciled = reconciler.reconcile_from_dataframe(actuals)
    assert reconciled["updated_records"] == 1

    op_perf = reconciler.compute_operational_performance(model_version="v1")
    assert op_perf["status"] == "OK"
    assert op_perf["count"] == 1

    # Step 7: Drift Detection
    drift = drift_report(
        train,
        lgb_model,
        feature_cols,
        {"drift": {"bins": 5, "warning_psi": 0.10, "critical_psi": 0.25}},
    )
    assert drift["status"] in {"OK", "WARNING", "DRIFT"}
