"""
Step 4: LightGBM Model Training for Demand Forecasting

This script trains a LightGBM regressor on the engineered features
and compares performance against the 7-Day MA baseline.

Key Rules:
- Same train/validation split as baseline
- Same metrics computation (per item-store, then average)
- Must beat baseline by 5-10% to proceed
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
FEATURES_PATH = BASE_DIR / "features" / "features_v2.parquet"  # Using enhanced features
MODEL_DIR = BASE_DIR / "registry" / "lightgbm_v1"
BASELINE_METRICS_PATH = BASE_DIR / "training" / "baseline_results" / "baseline_metrics.csv"

# Train/Validation split dates (same as baseline)
TRAIN_END = "2016-03-27"
VAL_START = "2016-03-28"
VAL_END = "2016-04-24"

# Features to use (exclude identifiers)
EXCLUDE_COLS = ['item_id', 'store_id', 'date', 'target']

# Baseline metrics to beat
BASELINE_7DAY_MA = {
    'MAE': 1.057,
    'RMSE': 1.401,
    'MAPE': 67.14
}

# Improvement thresholds
MIN_IMPROVEMENT_PCT = 5.0  # Must improve by at least 5%


def load_data() -> pd.DataFrame:
    """Load the feature dataset."""
    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df):,} rows")
    return df


def prepare_train_val_split(df: pd.DataFrame) -> tuple:
    """
    Split data into train and validation sets.
    
    Train: up to 2016-03-27
    Validate: 2016-03-28 → 2016-04-24
    """
    # Drop rows with NaN in features (from lag/rolling computations)
    df_clean = df.dropna()
    print(f"Rows after dropping NaN: {len(df_clean):,}")
    
    # Split by date
    train_df = df_clean[df_clean['date'] <= TRAIN_END].copy()
    val_df = df_clean[(df_clean['date'] >= VAL_START) & (df_clean['date'] <= VAL_END)].copy()
    
    print(f"\nTrain period: up to {TRAIN_END}")
    print(f"Train rows: {len(train_df):,}")
    print(f"\nValidation period: {VAL_START} to {VAL_END}")
    print(f"Validation rows: {len(val_df):,}")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    
    # Prepare X, y
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    
    # Keep metadata for per item-store evaluation
    val_meta = val_df[['item_id', 'store_id', 'date']].copy()
    
    return X_train, y_train, X_val, y_val, val_meta, feature_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
    """
    Train LightGBM regressor with proper early stopping.
    
    Using tuned parameters for demand forecasting.
    """
    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM MODEL")
    print("=" * 60)
    
    # LightGBM parameters (balanced for all metrics)
    params = {
        'objective': 'regression',      # MSE objective for balanced metrics
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 127,              # More complex trees
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
    
    print("\nModel parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Create and train model with validation set for early stopping
    model = lgb.LGBMRegressor(**params)
    
    print("\nTraining with early stopping on validation set...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae'
    )
    
    print(f"Best iteration: {model.best_iteration_}")
    
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    item_ids: np.ndarray, store_ids: np.ndarray) -> dict:
    """
    Compute metrics using same methodology as baseline:
    - Per item-store aggregation, then average
    - MAPE excludes y_true = 0
    """
    # Create DataFrame for groupby
    results_df = pd.DataFrame({
        'item_id': item_ids,
        'store_id': store_ids,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    def compute_group_metrics(group):
        yt = group['y_true'].values
        yp = group['y_pred'].values
        
        # MAE
        mae = np.mean(np.abs(yt - yp))
        
        # RMSE
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        
        # MAPE (exclude y_true = 0)
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


def evaluate_model(model: lgb.LGBMRegressor, X_val: pd.DataFrame, 
                   y_val: pd.Series, val_meta: pd.DataFrame) -> dict:
    """Evaluate model on validation set."""
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Clip predictions to non-negative (sales can't be negative)
    y_pred = np.clip(y_pred, 0, None)
    
    # Compute metrics
    metrics = compute_metrics(
        y_val.values, 
        y_pred, 
        val_meta['item_id'].values,
        val_meta['store_id'].values
    )
    
    print("\nLightGBM Model Metrics:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    return metrics


def compare_to_baseline(model_metrics: dict) -> tuple:
    """
    Compare model metrics to baseline.
    
    Returns (passed, comparison_dict)
    """
    print("\n" + "=" * 60)
    print("COMPARISON TO BASELINE (7-Day MA)")
    print("=" * 60)
    
    comparison = {}
    
    for metric in ['MAE', 'RMSE', 'MAPE']:
        baseline_val = BASELINE_7DAY_MA[metric]
        model_val = model_metrics[metric]
        
        # Lower is better for all metrics
        improvement = ((baseline_val - model_val) / baseline_val) * 100
        
        comparison[metric] = {
            'baseline': baseline_val,
            'model': model_val,
            'improvement_pct': improvement
        }
    
    # Print comparison table
    print("\n{:<10} {:<12} {:<12} {:<15}".format(
        "Metric", "Baseline", "LightGBM", "Improvement"
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
    
    # Check acceptance criteria
    mae_pass = comparison['MAE']['improvement_pct'] >= MIN_IMPROVEMENT_PCT
    rmse_pass = comparison['RMSE']['improvement_pct'] >= MIN_IMPROVEMENT_PCT
    no_degradation = all(comparison[m]['improvement_pct'] >= 0 for m in ['MAE', 'RMSE', 'MAPE'])
    
    passed = mae_pass and rmse_pass and no_degradation
    
    print(f"\nAcceptance Criteria (≥{MIN_IMPROVEMENT_PCT}% improvement):")
    print(f"  MAE improvement ≥ {MIN_IMPROVEMENT_PCT}%:  {'✓ PASS' if mae_pass else '✗ FAIL'}")
    print(f"  RMSE improvement ≥ {MIN_IMPROVEMENT_PCT}%: {'✓ PASS' if rmse_pass else '✗ FAIL'}")
    print(f"  No metric degradation:    {'✓ PASS' if no_degradation else '✗ FAIL'}")
    
    return passed, comparison


def get_feature_importance(model: lgb.LGBMRegressor, feature_cols: list) -> pd.DataFrame:
    """Get feature importance from trained model."""
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    
    for _, row in importance_df.iterrows():
        bar = '█' * int(row['importance'] / importance_df['importance'].max() * 20)
        print(f"  {row['feature']:<20} {row['importance']:>6} {bar}")
    
    return importance_df


def save_artifacts(model: lgb.LGBMRegressor, feature_cols: list, 
                   metrics: dict, comparison: dict, importance_df: pd.DataFrame) -> None:
    """Save all model artifacts to registry."""
    print("\n" + "=" * 60)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 60)
    
    # Create directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Save trained model
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # 2. Save feature list
    features_path = MODEL_DIR / "features.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  ✓ Feature list saved to {features_path}")
    
    # 3. Save training metadata
    metadata = {
        'model_type': 'LightGBM',
        'model_version': 'v1',
        'trained_at': datetime.now().isoformat(),
        'training_window': {
            'end': TRAIN_END
        },
        'validation_window': {
            'start': VAL_START,
            'end': VAL_END
        },
        'n_features': len(feature_cols),
        'best_iteration': model.best_iteration_,
        'params': model.get_params()
    }
    
    metadata_path = MODEL_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ Metadata saved to {metadata_path}")
    
    # 4. Save metrics
    metrics_data = {
        'model_metrics': metrics,
        'baseline_metrics': BASELINE_7DAY_MA,
        'comparison': comparison
    }
    
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  ✓ Metrics saved to {metrics_path}")
    
    # 5. Save feature importance
    importance_path = MODEL_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"  ✓ Feature importance saved to {importance_path}")
    
    print(f"\nAll artifacts saved to: {MODEL_DIR}")


def main():
    print("=" * 60)
    print("STEP 4: LIGHTGBM MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare train/val split
    X_train, y_train, X_val, y_val, val_meta, feature_cols = prepare_train_val_split(df)
    
    # Train model (pass validation set for early stopping)
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val, val_meta)
    
    # Compare to baseline
    passed, comparison = compare_to_baseline(metrics)
    
    # Feature importance
    importance_df = get_feature_importance(model, feature_cols)
    
    # Final verdict
    print("\n" + "=" * 60)
    if passed:
        print("✅ MODEL APPROVED - Beats baseline by required margin!")
        print("=" * 60)
        
        # Save artifacts
        save_artifacts(model, feature_cols, metrics, comparison, importance_df)
        
        print("\nReady for: Model Registry & Deployment Pipeline")
    else:
        print("❌ MODEL NOT APPROVED - Does not meet acceptance criteria")
        print("=" * 60)
        print("\nAction required:")
        print("  - Investigate feature engineering")
        print("  - Check for data issues")
        print("  - Do NOT deploy this model")
        
        # Still save for analysis
        save_artifacts(model, feature_cols, metrics, comparison, importance_df)
        print("\n(Artifacts saved for analysis)")
    
    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
