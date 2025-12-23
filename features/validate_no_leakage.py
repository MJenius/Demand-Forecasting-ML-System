"""
Step 3C: Validate No Leakage in Feature Pipeline

This script validates that the feature pipeline has no leakage by:
1. Reconstructing the 7-Day MA baseline from rolling_mean_7 feature
2. Comparing metrics against the original baseline results
3. Confirming they match within strict tolerance

Acceptance Criteria:
- MAE difference ≤ 0.01
- RMSE difference ≤ 0.01  
- MAPE difference ≤ 1%
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
FEATURES_PATH = BASE_DIR / "features" / "features.parquet"
BASELINE_METRICS_PATH = BASE_DIR / "training" / "baseline_results" / "baseline_metrics.csv"

# Validation window (same as baseline)
VAL_START = "2016-03-28"
VAL_END = "2016-04-24"

# Acceptance tolerances
MAE_TOL = 0.01
RMSE_TOL = 0.01
MAPE_TOL = 1.0  # percentage points


def load_features() -> pd.DataFrame:
    """Load the feature dataset."""
    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded {len(df):,} rows")
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by item_id, store_id, date
    df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)
    
    return df


def load_baseline_metrics() -> dict:
    """Load the original baseline metrics."""
    print(f"\nLoading baseline metrics from {BASELINE_METRICS_PATH}...")
    baseline_df = pd.read_csv(BASELINE_METRICS_PATH, index_col=0)
    
    # Extract 7-Day MA metrics
    ma7_row = baseline_df.loc['7-Day MA']
    
    metrics = {
        'MAE': ma7_row['MAE'],
        'RMSE': ma7_row['RMSE'],
        'MAPE': ma7_row['MAPE']
    }
    
    print(f"Original 7-Day MA baseline metrics:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    return metrics


def compute_metrics_from_features(df: pd.DataFrame) -> dict:
    """
    Reconstruct 7-Day MA baseline from rolling_mean_7 feature.
    
    Uses the same methodology as baseline.py:
    - Per item-store aggregation, then average
    - MAPE excludes y_true = 0
    """
    print(f"\n--- Reconstructing 7-Day MA from Features ---")
    
    # Filter to validation window
    val_df = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END)].copy()
    print(f"Validation window: {VAL_START} to {VAL_END}")
    print(f"Rows in validation window: {len(val_df):,}")
    
    # Drop rows where rolling_mean_7 is NaN
    val_df = val_df.dropna(subset=['rolling_mean_7'])
    print(f"Rows after dropping NaN rolling_mean_7: {len(val_df):,}")
    
    # Use rolling_mean_7 as prediction
    val_df['y_pred'] = val_df['rolling_mean_7']
    val_df['y_true'] = val_df['target']
    
    # Compute per item-store metrics
    def compute_group_metrics(group):
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE (exclude y_true = 0)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        return pd.Series({'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
    
    print("\nComputing per item-store metrics...")
    group_metrics = val_df.groupby(['item_id', 'store_id']).apply(
        compute_group_metrics, include_groups=False
    )
    
    # Average across all item-stores
    metrics = {
        'MAE': group_metrics['MAE'].mean(),
        'RMSE': group_metrics['RMSE'].mean(),
        'MAPE': group_metrics['MAPE'].mean()
    }
    
    print(f"\nReconstructed 7-Day MA metrics from features:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    return metrics


def validate_metrics(original: dict, reconstructed: dict) -> bool:
    """
    Validate that reconstructed metrics match original within tolerance.
    
    Returns True if validation passes, False otherwise.
    """
    print("\n" + "=" * 60)
    print("LEAKAGE VALIDATION RESULTS")
    print("=" * 60)
    
    # Compute differences
    mae_diff = abs(original['MAE'] - reconstructed['MAE'])
    rmse_diff = abs(original['RMSE'] - reconstructed['RMSE'])
    mape_diff = abs(original['MAPE'] - reconstructed['MAPE'])
    
    # Check each metric
    mae_pass = mae_diff <= MAE_TOL
    rmse_pass = rmse_diff <= RMSE_TOL
    mape_pass = mape_diff <= MAPE_TOL
    
    all_pass = mae_pass and rmse_pass and mape_pass
    
    # Print comparison table
    print("\n{:<12} {:<12} {:<12} {:<12} {:<8}".format(
        "Metric", "Original", "Features", "Diff", "Status"
    ))
    print("-" * 56)
    
    print("{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<8}".format(
        "MAE", original['MAE'], reconstructed['MAE'], mae_diff,
        "✓ PASS" if mae_pass else "✗ FAIL"
    ))
    
    print("{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<8}".format(
        "RMSE", original['RMSE'], reconstructed['RMSE'], rmse_diff,
        "✓ PASS" if rmse_pass else "✗ FAIL"
    ))
    
    print("{:<12} {:<12.2f}% {:<11.2f}% {:<12.2f} {:<8}".format(
        "MAPE", original['MAPE'], reconstructed['MAPE'], mape_diff,
        "✓ PASS" if mape_pass else "✗ FAIL"
    ))
    
    print("-" * 56)
    
    # Print tolerances
    print(f"\nTolerances: MAE ≤ {MAE_TOL}, RMSE ≤ {RMSE_TOL}, MAPE ≤ {MAPE_TOL}%")
    
    return all_pass


def main():
    print("=" * 60)
    print("STEP 3C: VALIDATE NO LEAKAGE IN FEATURE PIPELINE")
    print("=" * 60)
    
    # Load data
    features_df = load_features()
    original_metrics = load_baseline_metrics()
    
    # Reconstruct baseline from features
    reconstructed_metrics = compute_metrics_from_features(features_df)
    
    # Validate
    passed = validate_metrics(original_metrics, reconstructed_metrics)
    
    print("\n" + "=" * 60)
    if passed:
        print("✅ VALIDATION PASSED - Feature Pipeline Approved!")
        print("=" * 60)
        print("\nNo leakage detected. You are cleared for STEP 4: ML Model Training")
        print("\nCompleted steps:")
        print("  ✓ Step 1: Data ingestion")
        print("  ✓ Step 2: Baselines")
        print("  ✓ Step 3: Feature pipeline (production-safe)")
    else:
        print("❌ VALIDATION FAILED - Leakage Detected!")
        print("=" * 60)
        print("\nSTOP: Do not proceed to ML training.")
        print("Fix the feature pipeline before continuing.")
    
    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
