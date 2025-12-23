"""
Step 4B: Two-Model Strategy with SKU Segmentation

Strategy:
- Train LightGBM on normal-volume SKUs (mean sales >= 1.5)
- Use 7-Day MA for low-volume SKUs (mean sales < 1.5)
- Combine predictions at inference time

This improves MAE by avoiding over-modeling noise in low-volume items.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
FEATURES_PATH = BASE_DIR / "features" / "features_v2.parquet"
SKU_SEGMENTS_PATH = BASE_DIR / "training" / "sku_segments.csv"
MODEL_DIR = BASE_DIR / "registry" / "lightgbm_segmented"
BASELINE_METRICS_PATH = BASE_DIR / "training" / "baseline_results" / "baseline_metrics.csv"

# Split config
TRAIN_END = "2016-03-27"
VAL_START = "2016-03-28"
VAL_END = "2016-04-24"

# Segmentation threshold
VOLUME_THRESHOLD = 1.5

EXCLUDE_COLS = ['item_id', 'store_id', 'date', 'target']
BASELINE_7DAY_MA = {
    'MAE': 1.057,
    'RMSE': 1.401,
    'MAPE': 67.14
}
MIN_IMPROVEMENT_PCT = 5.0


def load_data() -> pd.DataFrame:
    """Load feature dataset."""
    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df):,} rows")
    return df


def load_sku_segments() -> pd.DataFrame:
    """Load pre-computed SKU mean sales for segmentation."""
    print(f"Loading SKU segments from {SKU_SEGMENTS_PATH}...")
    segments = pd.read_csv(SKU_SEGMENTS_PATH)
    print(f"Loaded {len(segments):,} SKUs")
    return segments


def segment_data(df: pd.DataFrame, segments: pd.DataFrame) -> tuple:
    """
    Segment data into low-volume and normal-volume.
    
    Returns (df_normal, df_low, normal_sku_list)
    """
    print(f"\n--- SKU SEGMENTATION ---")
    print(f"Threshold: mean sales >= {VOLUME_THRESHOLD}")
    
    # Add volume segment to segments
    segments['is_normal_volume'] = segments['mean_sales'] >= VOLUME_THRESHOLD
    
    # Get SKU lists
    normal_skus = segments[segments['is_normal_volume']][['item_id', 'store_id']]
    low_skus = segments[~segments['is_normal_volume']][['item_id', 'store_id']]
    
    print(f"\nNormal-volume SKUs: {len(normal_skus):,} ({len(normal_skus)/len(segments)*100:.1f}%)")
    print(f"Low-volume SKUs: {len(low_skus):,} ({len(low_skus)/len(segments)*100:.1f}%)")
    
    # Create merge keys
    normal_skus['merge_key'] = normal_skus['item_id'] + '_' + normal_skus['store_id']
    low_skus['merge_key'] = low_skus['item_id'] + '_' + low_skus['store_id']
    df['merge_key'] = df['item_id'] + '_' + df['store_id']
    
    # Segment data
    df_normal = df[df['merge_key'].isin(normal_skus['merge_key'])].copy()
    df_low = df[df['merge_key'].isin(low_skus['merge_key'])].copy()
    
    print(f"\nNormal-volume data: {len(df_normal):,} rows ({len(df_normal)/len(df)*100:.1f}%)")
    print(f"Low-volume data: {len(df_low):,} rows ({len(df_low)/len(df)*100:.1f}%)")
    
    return df_normal, df_low, normal_skus, low_skus, segments


def prepare_train_val(df: pd.DataFrame, include_all: bool = False) -> tuple:
    """Prepare train/val split for a segment."""
    df_clean = df.dropna(subset=['target'])
    
    # Drop non-numeric columns (like merge_key)
    df_clean = df_clean.drop(columns=['merge_key'], errors='ignore')
    
    # Split by date
    train_df = df_clean[df_clean['date'] <= TRAIN_END].copy()
    val_df = df_clean[(df_clean['date'] >= VAL_START) & (df_clean['date'] <= VAL_END)].copy()
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS and col != 'merge_key']
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    
    val_meta = val_df[['item_id', 'store_id', 'date']].copy()
    
    return X_train, y_train, X_val, y_val, val_meta, feature_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
    """Train LightGBM on normal-volume data."""
    print("\nTraining LightGBM on normal-volume SKUs...")
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': 10,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 15,
        'n_estimators': 1500,
        'early_stopping_rounds': 100,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae')
    
    print(f"Best iteration: {model.best_iteration_}")
    return model


def compute_7day_ma(df: pd.DataFrame) -> pd.Series:
    """Compute 7-day MA for a segment (for low-volume SKUs)."""
    df = df.copy()
    df['group_key'] = df['item_id'] + '_' + df['store_id']
    
    # Compute 7-day rolling mean (shifted to avoid leakage)
    rolling_ma = (
        df.groupby('group_key', sort=False)['target']
        .transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
    )
    
    return rolling_ma


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    item_ids: np.ndarray, store_ids: np.ndarray) -> dict:
    """Compute per-SKU then averaged metrics."""
    results_df = pd.DataFrame({
        'item_id': item_ids,
        'store_id': store_ids,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    def compute_group_metrics(group):
        yt = group['y_true'].values
        yp = group['y_pred'].values
        
        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        
        mask = yt != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100
        else:
            mape = np.nan
        
        return pd.Series({'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
    
    group_metrics = results_df.groupby(['item_id', 'store_id']).apply(
        compute_group_metrics, include_groups=False
    )
    
    metrics = {
        'MAE': group_metrics['MAE'].mean(),
        'RMSE': group_metrics['RMSE'].mean(),
        'MAPE': group_metrics['MAPE'].mean()
    }
    
    return metrics


def evaluate_segmented_model(model: lgb.LGBMRegressor, 
                            val_normal: pd.DataFrame,
                            val_low: pd.DataFrame,
                            feature_cols: list) -> dict:
    """Evaluate two-model strategy on validation set."""
    print("\n" + "=" * 60)
    print("EVALUATING TWO-MODEL STRATEGY")
    print("=" * 60)
    
    # For normal-volume: drop NaN and get predictions
    val_normal_clean = val_normal.dropna(subset=feature_cols + ['target'])
    val_normal_clean = val_normal_clean.drop(columns=['merge_key'], errors='ignore')
    
    print(f"\nNormal-volume validation: {len(val_normal_clean):,} rows")
    
    if len(val_normal_clean) > 0:
        y_pred_normal = model.predict(val_normal_clean[feature_cols])
        y_pred_normal = np.clip(y_pred_normal, 0, None)
        y_true_normal = val_normal_clean['target'].values
        item_ids_normal = val_normal_clean['item_id'].values
        store_ids_normal = val_normal_clean['store_id'].values
    else:
        y_pred_normal = np.array([])
        y_true_normal = np.array([])
        item_ids_normal = np.array([])
        store_ids_normal = np.array([])
    
    # For low-volume: compute 7-day MA
    val_low_clean = val_low.dropna(subset=['target'])
    val_low_clean = val_low_clean.drop(columns=['merge_key'], errors='ignore')
    val_low_clean['group_key'] = val_low_clean['item_id'] + '_' + val_low_clean['store_id']
    val_low_clean = val_low_clean.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)
    
    print(f"Low-volume validation: {len(val_low_clean):,} rows")
    
    if len(val_low_clean) > 0:
        # Compute 7-day MA (shifted)
        y_pred_low = (
            val_low_clean.groupby('group_key', sort=False)['target']
            .transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
        )
        y_pred_low = y_pred_low.fillna(method='bfill').fillna(method='ffill')
        y_true_low = val_low_clean['target'].values
        item_ids_low = val_low_clean['item_id'].values
        store_ids_low = val_low_clean['store_id'].values
    else:
        y_pred_low = np.array([])
        y_true_low = np.array([])
        item_ids_low = np.array([])
        store_ids_low = np.array([])
    
    # Combine all predictions
    all_true = np.concatenate([y_true_normal, y_true_low])
    all_pred = np.concatenate([y_pred_normal, y_pred_low])
    all_item_ids = np.concatenate([item_ids_normal, item_ids_low])
    all_store_ids = np.concatenate([store_ids_normal, store_ids_low])
    
    print(f"Total validation rows: {len(all_true):,}")
    
    # Compute metrics
    metrics = compute_metrics(all_true, all_pred, all_item_ids, all_store_ids)
    
    print(f"\nTwo-Model Strategy Metrics:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Breakdown by segment
    if len(y_true_normal) > 0:
        metrics_normal = compute_metrics(y_true_normal, y_pred_normal,
                                         item_ids_normal, store_ids_normal)
        print(f"\nNormal-volume (LightGBM): MAE={metrics_normal['MAE']:.4f}, RMSE={metrics_normal['RMSE']:.4f}")
    
    if len(y_true_low) > 0:
        metrics_low = compute_metrics(y_true_low, y_pred_low,
                                      item_ids_low, store_ids_low)
        print(f"Low-volume (7-day MA):    MAE={metrics_low['MAE']:.4f}, RMSE={metrics_low['RMSE']:.4f}")
    
    return metrics


def compare_to_baseline(metrics: dict) -> tuple:
    """Compare segmented model to baseline."""
    print("\n" + "=" * 60)
    print("COMPARISON TO BASELINE (7-Day MA)")
    print("=" * 60)
    
    comparison = {}
    for metric in ['MAE', 'RMSE', 'MAPE']:
        baseline_val = BASELINE_7DAY_MA[metric]
        model_val = metrics[metric]
        improvement = ((baseline_val - model_val) / baseline_val) * 100
        comparison[metric] = {
            'baseline': baseline_val,
            'model': model_val,
            'improvement_pct': improvement
        }
    
    print("\n{:<10} {:<12} {:<12} {:<15}".format(
        "Metric", "Baseline", "Two-Model", "Improvement"
    ))
    print("-" * 50)
    
    for metric in ['MAE', 'RMSE', 'MAPE']:
        c = comparison[metric]
        unit = '%' if metric == 'MAPE' else ''
        print("{:<10} {:<12.4f}{} {:<12.4f}{} {:>+.2f}%".format(
            metric,
            c['baseline'], unit,
            c['model'], unit,
            c['improvement_pct']
        ))
    
    print("-" * 50)
    
    mae_pass = comparison['MAE']['improvement_pct'] >= MIN_IMPROVEMENT_PCT
    rmse_pass = comparison['RMSE']['improvement_pct'] >= MIN_IMPROVEMENT_PCT
    no_degrade = all(comparison[m]['improvement_pct'] >= 0 for m in ['MAE', 'RMSE', 'MAPE'])
    
    passed = mae_pass and rmse_pass and no_degrade
    
    print(f"\nAcceptance Criteria (≥{MIN_IMPROVEMENT_PCT}% improvement):")
    print(f"  MAE improvement ≥ {MIN_IMPROVEMENT_PCT}%:  {'✓ PASS' if mae_pass else '✗ FAIL'}")
    print(f"  RMSE improvement ≥ {MIN_IMPROVEMENT_PCT}%: {'✓ PASS' if rmse_pass else '✗ FAIL'}")
    print(f"  No metric degradation:    {'✓ PASS' if no_degrade else '✗ FAIL'}")
    
    return passed, comparison


def save_artifacts(model: lgb.LGBMRegressor, feature_cols: list,
                   metrics: dict, comparison: dict, segments: pd.DataFrame) -> None:
    """Save model and metadata."""
    print("\n" + "=" * 60)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 60)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # Save features
    features_path = MODEL_DIR / "features.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  ✓ Features saved to {features_path}")
    
    # Save segmentation
    segments_path = MODEL_DIR / "sku_segments.csv"
    segments.to_csv(segments_path, index=False)
    print(f"  ✓ SKU segments saved to {segments_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'LightGBM + 7-Day MA (Segmented)',
        'model_version': 'v1-segmented',
        'trained_at': datetime.now().isoformat(),
        'strategy': 'Two-model: LightGBM for normal-volume, 7-day MA for low-volume',
        'volume_threshold': VOLUME_THRESHOLD,
        'training_window': {'end': TRAIN_END},
        'validation_window': {'start': VAL_START, 'end': VAL_END},
        'n_features': len(feature_cols),
        'best_iteration': model.best_iteration_,
        'params': model.get_params()
    }
    
    metadata_path = MODEL_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ Metadata saved to {metadata_path}")
    
    # Save metrics
    metrics_data = {
        'model_metrics': metrics,
        'baseline_metrics': BASELINE_7DAY_MA,
        'comparison': comparison
    }
    
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  ✓ Metrics saved to {metrics_path}")
    
    print(f"\nAll artifacts saved to: {MODEL_DIR}")


def main():
    print("=" * 60)
    print("STEP 4B: TWO-MODEL STRATEGY WITH SKU SEGMENTATION")
    print("=" * 60)
    
    # Load data
    df = load_data()
    segments = load_sku_segments()
    
    # Segment
    df_normal, df_low, normal_skus, low_skus, segments = segment_data(df, segments)
    
    # Prepare train/val for normal-volume
    print("\nPreparing training data (normal-volume SKUs)...")
    X_train, y_train, X_val, y_val, val_meta, feature_cols = prepare_train_val(df_normal)
    print(f"Train rows: {len(X_train):,}, Val rows: {len(X_val):,}")
    print(f"Features: {len(feature_cols)}")
    
    # Train LightGBM on normal-volume
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate segmented strategy
    metrics = evaluate_segmented_model(model, df_normal, df_low, feature_cols)
    
    # Compare
    passed, comparison = compare_to_baseline(metrics)
    
    # Final verdict
    print("\n" + "=" * 60)
    if passed:
        print("✅ TWO-MODEL STRATEGY APPROVED!")
        print("=" * 60)
        save_artifacts(model, feature_cols, metrics, comparison, segments)
        print("\nReady for: Model Registry & Deployment")
    else:
        print("❌ STRATEGY NOT APPROVED")
        print("=" * 60)
        save_artifacts(model, feature_cols, metrics, comparison, segments)
        print("\n(Artifacts saved for analysis)")
    
    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
