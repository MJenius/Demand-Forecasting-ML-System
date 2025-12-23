"""
Baseline Forecasting Models for Demand Forecasting

Implements:
1. Last-Value Forecast: ŷ(t) = sales(t-1)
2. 7-Day Moving Average: ŷ(t) = mean(sales[t-7:t-1])
3. Seasonal Naive: ŷ(t) = sales(t-7)

Evaluation:
- Time-based split: last 28 days as validation
- Metrics computed per item-store, then averaged
- MAE, RMSE, MAPE (excluding zeros)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Use a sample for faster execution (set to None for full data)
# For development: 100-500 items is fast. For production: None (all ~3049 items)
# Note: 500 items = ~16% sample which gives statistically valid results
SAMPLE_ITEMS = 500  # Number of item_ids to sample (None = all)


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    """Load data and sort by (item_id, store_id, date)."""
    print(f"Loading data from {data_path}...")
    
    # Read only needed columns
    df = pd.read_parquet(
        data_path, 
        engine='pyarrow',
        columns=['date', 'item_id', 'store_id', 'sales']
    )
    print(f"Loaded {len(df):,} rows")
    
    # Sample items if requested (for faster iteration)
    if SAMPLE_ITEMS is not None:
        unique_items = df['item_id'].unique()
        np.random.seed(42)
        sampled_items = np.random.choice(unique_items, min(SAMPLE_ITEMS, len(unique_items)), replace=False)
        df = df[df['item_id'].isin(sampled_items)]
        print(f"Sampled {SAMPLE_ITEMS} items: {len(df):,} rows")
        
        # Only sort if sampling (smaller dataset)
        print("Sorting by (item_id, store_id, date)...")
        df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)
    else:
        # Data was already sorted in prepare_timeseries.py
        # Just ensure it's sorted by date within groups (already is)
        print("Using pre-sorted data from parquet file...")
    
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique items: {df['item_id'].nunique()}, Unique stores: {df['store_id'].nunique()}")
    
    return df


def get_validation_split(df: pd.DataFrame, val_days: int = 28):
    """Split into train/validation using last N days."""
    max_date = df['date'].max()
    val_start = max_date - pd.Timedelta(days=val_days - 1)
    
    print(f"\nValidation window: {val_start.date()} → {max_date.date()}")
    
    return val_start


def compute_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all baseline predictions using pandas groupby + transform.
    
    For each (item_id, store_id) group:
    - last_value: shift(1)
    - ma_7day: rolling(7).mean().shift(1)
    - seasonal_naive: shift(7)
    """
    print("\nComputing baseline predictions...")
    
    df = df.copy()
    
    # Create group identifier
    df['group_key'] = df['item_id'] + '_' + df['store_id']
    
    print("Computing Last Value predictions...")
    df['pred_last_value'] = df.groupby('group_key', sort=False)['sales'].shift(1)
    
    print("Computing Seasonal Naive predictions...")
    df['pred_seasonal'] = df.groupby('group_key', sort=False)['sales'].shift(7)
    
    print("Computing 7-Day MA predictions...")
    df['pred_ma_7day'] = (
        df.groupby('group_key', sort=False)['sales']
        .transform(lambda x: x.rolling(7, min_periods=7).mean().shift(1))
    )
    
    print("Predictions computed.")
    return df


def compute_metrics_per_group(df: pd.DataFrame, pred_col: str) -> dict:
    """
    Compute metrics per item-store group, then average.
    
    This avoids dominance by high-volume items.
    """
    # Filter to rows with valid predictions
    valid = df.dropna(subset=[pred_col])
    
    if len(valid) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
    
    # Compute metrics per group using group_key
    group_metrics = []
    
    for name, group in valid.groupby('group_key', sort=False):
        y_true = group['sales'].values.astype(float)
        y_pred = group[pred_col].values.astype(float)
        
        if len(y_true) == 0:
            continue
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE (exclude zeros)
        nonzero = y_true > 0
        if nonzero.sum() > 0:
            mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        else:
            mape = np.nan
        
        group_metrics.append({'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
    
    # Average across groups
    metrics_df = pd.DataFrame(group_metrics)
    
    return {
        'MAE': metrics_df['MAE'].mean(),
        'RMSE': metrics_df['RMSE'].mean(),
        'MAPE': metrics_df['MAPE'].mean()
    }


def evaluate_baselines(df: pd.DataFrame, val_start: pd.Timestamp) -> dict:
    """Evaluate all baselines on validation data."""
    # Filter to validation window
    val_df = df[df['date'] >= val_start].copy()
    print(f"\nValidation set: {len(val_df):,} rows")
    
    results = {}
    
    # Baseline 1: Last Value
    print("\nEvaluating Baseline 1: Last Value...")
    results['Last Value'] = compute_metrics_per_group(val_df, 'pred_last_value')
    
    # Baseline 2: 7-Day Moving Average
    print("Evaluating Baseline 2: 7-Day MA...")
    results['7-Day MA'] = compute_metrics_per_group(val_df, 'pred_ma_7day')
    
    # Baseline 3: Seasonal Naive
    print("Evaluating Baseline 3: Seasonal Naive...")
    results['Seasonal Naive'] = compute_metrics_per_group(val_df, 'pred_seasonal')
    
    return results


def print_results(results: dict, val_start, val_end):
    """Print formatted results."""
    print("\n" + "="*60)
    print("BASELINE FORECASTING RESULTS")
    print("="*60)
    print(f"\nValidation window: {val_start.date()} → {val_end.date()}")
    print()
    
    for name, metrics in results.items():
        print(f"Baseline: {name}")
        print(f"  MAE:  {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.1f}%")
        print()
    
    print("="*60)


def main():
    """Run baseline evaluation."""
    # Paths
    data_path = Path(__file__).parent.parent / "data" / "processed" / "sales_long.parquet"
    
    # Load and prepare data
    df = load_and_prepare_data(data_path)
    
    # Get validation split
    val_start = get_validation_split(df, val_days=28)
    val_end = df['date'].max()
    
    # Compute predictions
    df = compute_baseline_predictions(df)
    
    # Evaluate
    results = evaluate_baselines(df, val_start)
    
    # Print results
    print_results(results, val_start, val_end)
    
    # Save results
    output_dir = Path(__file__).parent / "baseline_results"
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "baseline_metrics.csv")
    print(f"Results saved to {output_dir / 'baseline_metrics.csv'}")
    
    # Sanity checks
    print("\n--- Sanity Checks ---")
    print(f"✓ All metrics are finite: {all(np.isfinite(m['MAE']) for m in results.values())}")
    print(f"✓ Baselines differ: {results['Last Value']['MAE'] != results['7-Day MA']['MAE']}")
    

if __name__ == "__main__":
    main()
