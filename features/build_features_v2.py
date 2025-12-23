"""
Enhanced Feature Engineering Pipeline v2 for Demand Forecasting

Adds to base features:
- Price features (from sell_prices.csv)
- Longer lag features (lag_21, lag_28)
- More rolling windows (rolling_mean_28)
- Price momentum features

This version aims to improve model performance to meet the 5% improvement threshold.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sampling for faster processing (same as v1 for comparison)
SAMPLE_ITEMS = 500


def load_sales_data(data_dir: Path) -> pd.DataFrame:
    """Load the long-format sales data."""
    sales_path = data_dir / "data" / "processed" / "sales_long.parquet"
    print(f"Loading sales data from {sales_path}...")
    
    df = pd.read_parquet(sales_path, engine='pyarrow')
    print(f"Loaded {len(df):,} rows")
    
    # Sample for faster processing
    if SAMPLE_ITEMS is not None:
        unique_items = df['item_id'].unique()
        np.random.seed(42)  # Same seed as baseline
        sampled_items = np.random.choice(unique_items, min(SAMPLE_ITEMS, len(unique_items)), replace=False)
        df = df[df['item_id'].isin(sampled_items)]
        print(f"Sampled {SAMPLE_ITEMS} items: {len(df):,} rows")
    
    # Sort by item_id, store_id, date
    df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)
    
    return df


def load_calendar_data(data_dir: Path) -> pd.DataFrame:
    """Load and process calendar data for event features and wm_yr_wk."""
    calendar_path = data_dir / "data" / "raw" / "m5" / "calendar.csv"
    print(f"Loading calendar data from {calendar_path}...")
    
    calendar = pd.read_csv(calendar_path)
    calendar['date'] = pd.to_datetime(calendar['date'])
    
    # Create event indicator
    calendar['is_event'] = (
        calendar['event_name_1'].notna() | 
        calendar['event_name_2'].notna()
    ).astype(int)
    
    # Keep wm_yr_wk for price join
    calendar = calendar[['date', 'wm_yr_wk', 'is_event']].drop_duplicates()
    
    print(f"Loaded calendar with {len(calendar)} dates, {calendar['is_event'].sum()} event days")
    
    return calendar


def load_price_data(data_dir: Path, sampled_items: list) -> pd.DataFrame:
    """Load sell prices filtered to sampled items."""
    prices_path = data_dir / "data" / "raw" / "m5" / "sell_prices.csv"
    print(f"Loading price data from {prices_path}...")
    
    prices = pd.read_csv(prices_path)
    
    # Filter to sampled items
    if sampled_items is not None:
        prices = prices[prices['item_id'].isin(sampled_items)]
    
    print(f"Loaded {len(prices):,} price records")
    
    return prices


def merge_prices(df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Merge price data with sales data using wm_yr_wk."""
    print("\nMerging price data...")
    
    # Add wm_yr_wk to sales data via calendar
    df = df.merge(calendar[['date', 'wm_yr_wk']], on='date', how='left')
    
    # Merge prices
    df = df.merge(
        prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']], 
        on=['store_id', 'item_id', 'wm_yr_wk'], 
        how='left'
    )
    
    # Fill missing prices with forward fill within each item-store
    df['sell_price'] = df.groupby(['item_id', 'store_id'])['sell_price'].ffill()
    df['sell_price'] = df.groupby(['item_id', 'store_id'])['sell_price'].bfill()
    
    # Drop wm_yr_wk (no longer needed)
    df = df.drop(columns=['wm_yr_wk'])
    
    print(f"Price coverage: {df['sell_price'].notna().sum()/len(df)*100:.1f}%")
    
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features - extended set for better performance."""
    print("\nCreating lag features...")
    
    df = df.copy()
    df['group_key'] = df['item_id'] + '_' + df['store_id']
    
    lags = [1, 7, 14, 21, 28]  # Extended lag set
    for lag in lags:
        print(f"  - lag_{lag}")
        df[f'lag_{lag}'] = df.groupby('group_key', sort=False)['sales'].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling statistics - extended set."""
    print("\nCreating rolling features (shifted to avoid leakage)...")
    
    windows = [7, 14, 28]
    
    for window in windows:
        # Rolling mean
        print(f"  - rolling_mean_{window}")
        df[f'rolling_mean_{window}'] = (
            df.groupby('group_key', sort=False)['sales']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        )
        
        # Rolling std
        print(f"  - rolling_std_{window}")
        df[f'rolling_std_{window}'] = (
            df.groupby('group_key', sort=False)['sales']
            .transform(lambda x: x.rolling(window, min_periods=2).std().shift(1))
        )
    
    # Rolling max and min (7-day)
    print("  - rolling_max_7")
    df['rolling_max_7'] = (
        df.groupby('group_key', sort=False)['sales']
        .transform(lambda x: x.rolling(7, min_periods=1).max().shift(1))
    )
    
    print("  - rolling_min_7")
    df['rolling_min_7'] = (
        df.groupby('group_key', sort=False)['sales']
        .transform(lambda x: x.rolling(7, min_periods=1).min().shift(1))
    )
    
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price-based features."""
    print("\nCreating price features...")
    
    # Current price
    print("  - sell_price")
    
    # Price relative to item's historical average (price index)
    print("  - price_norm")
    df['price_norm'] = df.groupby(['item_id', 'store_id'])['sell_price'].transform(
        lambda x: x / x.mean()
    )
    
    # Price change from previous week
    print("  - price_change")
    df['price_change'] = df.groupby('group_key', sort=False)['sell_price'].diff(7).fillna(0)
    
    # Is price reduced? (potential sale indicator)
    print("  - is_price_reduced")
    df['is_price_reduced'] = (df['price_change'] < 0).astype(int)
    
    return df


def create_calendar_features(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features."""
    print("\nCreating calendar features...")
    
    # Day of week
    print("  - day_of_week")
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Week of year
    print("  - week_of_year")
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Is weekend
    print("  - is_weekend")
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Month
    print("  - month")
    df['month'] = df['date'].dt.month
    
    # Day of month
    print("  - day_of_month")
    df['day_of_month'] = df['date'].dt.day
    
    # Is month end (last 5 days)
    print("  - is_month_end")
    df['is_month_end'] = (df['day_of_month'] >= 26).astype(int)
    
    # Is month start (first 5 days)
    print("  - is_month_start")
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    
    # Is event (from calendar)
    print("  - is_event")
    df = df.merge(calendar[['date', 'is_event']], on='date', how='left')
    df['is_event'] = df['is_event'].fillna(0).astype(int)
    
    return df


def build_feature_dataframe(df: pd.DataFrame) -> tuple:
    """Build final feature dataframe with target."""
    print("\nBuilding final feature dataframe...")
    
    # Define feature columns (everything except identifiers and target)
    all_features = [
        # Lags
        'lag_1', 'lag_7', 'lag_14', 'lag_21', 'lag_28',
        # Rolling
        'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
        'rolling_std_7', 'rolling_std_14', 'rolling_std_28',
        'rolling_max_7', 'rolling_min_7',
        # Price
        'sell_price', 'price_norm', 'price_change', 'is_price_reduced',
        # Calendar
        'day_of_week', 'week_of_year', 'is_weekend', 'month',
        'day_of_month', 'is_month_end', 'is_month_start', 'is_event'
    ]
    
    # Keep only columns that exist
    existing_features = [f for f in all_features if f in df.columns]
    
    # Define output columns
    output_cols = ['item_id', 'store_id', 'date', 'target'] + existing_features
    
    # Rename sales to target
    df = df.rename(columns={'sales': 'target'})
    
    # Drop helper columns
    df = df.drop(columns=['group_key'], errors='ignore')
    
    # Select and reorder columns
    df = df[[col for col in output_cols if col in df.columns]]
    
    return df, existing_features


def validate_no_leakage(df: pd.DataFrame) -> None:
    """Validate no leakage in features."""
    print("\n--- Leakage Validation ---")
    
    first_item = df['item_id'].iloc[0]
    first_store = df['store_id'].iloc[0]
    sample = df[(df['item_id'] == first_item) & (df['store_id'] == first_store)].copy()
    sample = sample.sort_values('date').reset_index(drop=True)
    
    # Check lag_1
    lag1_correct = (sample['lag_1'].iloc[1:].values == sample['target'].iloc[:-1].values).mean()
    print(f"✓ lag_1 correctly shifted: {lag1_correct*100:.1f}% match")
    
    # Check lag_7
    lag7_correct = (sample['lag_7'].iloc[7:].values == sample['target'].iloc[:-7].values).mean()
    print(f"✓ lag_7 correctly shifted: {lag7_correct*100:.1f}% match")
    
    # Check rolling_mean_7
    expected_roll = sample['target'].iloc[0:7].mean()
    actual_roll = sample['rolling_mean_7'].iloc[7]
    roll_correct = np.isclose(expected_roll, actual_roll)
    print(f"✓ rolling_mean_7 correctly shifted: {roll_correct}")
    
    if lag1_correct > 0.99 and lag7_correct > 0.99 and roll_correct:
        print("✓ No leakage detected!")
    else:
        print("⚠ WARNING: Possible leakage!")


def main():
    """Build enhanced feature pipeline v2."""
    project_dir = Path(__file__).parent.parent
    
    # Load data
    df = load_sales_data(project_dir)
    calendar = load_calendar_data(project_dir)
    
    # Get sampled items for price filtering
    sampled_items = df['item_id'].unique().tolist()
    
    # Load and merge prices
    prices = load_price_data(project_dir, sampled_items)
    df = merge_prices(df, calendar, prices)
    
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique items: {df['item_id'].nunique()}, Unique stores: {df['store_id'].nunique()}")
    
    # Create features
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_price_features(df)
    df = create_calendar_features(df, calendar)
    
    # Build final dataframe
    df, feature_list = build_feature_dataframe(df)
    
    # Validate
    validate_no_leakage(df)
    
    # Find earliest usable date
    feature_cols = [col for col in df.columns if col not in ['item_id', 'store_id', 'date', 'target']]
    df_complete = df.dropna(subset=feature_cols)
    earliest_date = df_complete['date'].min()
    print(f"\nEarliest usable date (after lags): {earliest_date.date()}")
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY (v2 - Enhanced)")
    print("="*60)
    print(f"\nFeatures created ({len(feature_list)}):")
    for f in feature_list:
        print(f"  - {f}")
    
    print(f"\nDataframe shape: {df.shape}")
    print(f"Total rows: {len(df):,}")
    print(f"Complete rows (no NaN): {len(df_complete):,}")
    
    # Save
    output_path = project_dir / "features" / "features_v2.parquet"
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df):,} rows")
    
    print("\n" + "="*60)
    print("Feature pipeline v2 complete!")
    print("="*60)
    
    return df, feature_list


if __name__ == "__main__":
    df, features = main()
