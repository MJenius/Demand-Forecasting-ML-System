"""Enterprise forecasting evaluation metrics suite.

Provides standardized, zero-safe implementations for:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error)
- Bias (Forecast Bias: sum(pred - actual) / sum(actual))
- MAPE (Mean Absolute Percentage Error on non-zero actuals)
- Granular stratified metrics by SKU volume tier, horizon step, and store
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_forecasts(
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Calculate point metrics across ground truth and predictions.

    Args:
        y_true: Observed ground-truth demand values.
        y_pred: Forecasted demand values (clipped >= 0).
        epsilon: Small epsilon to prevent division by zero in WAPE and Bias.

    Returns:
        dict containing MAE, RMSE, WAPE, Bias, and MAPE.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    error = actual - predicted
    sum_actual = float(np.sum(actual))
    nonzero = actual > 0

    mae = float(np.mean(np.abs(error))) if len(error) > 0 else 0.0
    rmse = float(np.sqrt(np.mean(error**2))) if len(error) > 0 else 0.0

    denominator = max(sum_actual, epsilon)
    wape = float(np.sum(np.abs(error)) / denominator)
    bias = float(np.sum(predicted - actual) / denominator)

    if nonzero.any():
        mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
    else:
        mape = float("nan")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "Bias": bias,
        "MAPE": mape,
    }


def evaluate_stratified_slices(
    df: pd.DataFrame,
    y_true_col: str = "target",
    y_pred_col: str = "prediction",
    item_col: str = "item_id",
    store_col: str = "store_id",
    horizon_col: str | None = "horizon",
    mean_sales_col: str | None = "mean_sales",
) -> dict[str, list[dict]]:
    """Compute stratified forecasting performance by horizon, store, and volume tier.

    Volume tiers:
    - intermittent: mean_sales < 0.5
    - low: 0.5 <= mean_sales < 1.5
    - normal: mean_sales >= 1.5
    """
    results: dict[str, list[dict]] = {}

    # 1. By Horizon
    if horizon_col and horizon_col in df.columns:
        horizon_rows = []
        for h, group in df.groupby(horizon_col):
            m = evaluate_forecasts(group[y_true_col], group[y_pred_col])
            horizon_rows.append({"horizon": int(h), "count": len(group), **m})
        results["by_horizon"] = sorted(horizon_rows, key=lambda x: x["horizon"])

    # 2. By Store
    if store_col in df.columns:
        store_rows = []
        for s, group in df.groupby(store_col):
            m = evaluate_forecasts(group[y_true_col], group[y_pred_col])
            store_rows.append({"store_id": str(s), "count": len(group), **m})
        results["by_store"] = sorted(store_rows, key=lambda x: x["store_id"])

    # 3. By Volume Tier
    if mean_sales_col and mean_sales_col in df.columns:
        tier_col = pd.Series("normal", index=df.index)
        tier_col[df[mean_sales_col] < 1.5] = "low"
        tier_col[df[mean_sales_col] < 0.5] = "intermittent"

        tier_rows = []
        for tier, group in df.groupby(tier_col):
            m = evaluate_forecasts(group[y_true_col], group[y_pred_col])
            tier_rows.append({"volume_tier": str(tier), "count": len(group), **m})
        results["by_volume_tier"] = tier_rows

    return results
