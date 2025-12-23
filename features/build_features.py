"""
Feature Engineering Pipeline for Demand Forecasting

Creates deterministic, reusable features that:
- Work for training and inference
- Avoid leakage (all features shifted properly)
- Support retraining and drift detection

Feature Groups:
1. Lag features: lag_1, lag_7, lag_14
2. Rolling statistics (shifted): 7-day mean, 14-day mean, 7-day std
3. Calendar features: day_of_week, week_of_year, is_weekend, is_event

Output: features/features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sampling for faster processing (set to None for full data)
SAMPLE_ITEMS = 500  # Use same sample as baseline for consistency


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
    """Load and process calendar data for event features."""
    calendar_path = data_dir / "data" / "raw" / "m5" / "calendar.csv"
    print(f"Loading calendar data from {calendar_path}...")
    
    calendar = pd.read_csv(calendar_path)
    calendar['date'] = pd.to_datetime(calendar['date'])
    
    # Create event indicator (any event on that day)
    calendar['is_event'] = (
        calendar['event_name_1'].notna() | 
        calendar['event_name_2'].notna()
    ).astype(int)
    
    # Keep only relevant columns
    calendar = calendar[['date', 'is_event']].drop_duplicates()
    
    print(f"Loaded calendar with {len(calendar)} dates, {calendar['is_event'].sum()} event days")
    
    return calendar


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features for each item-store group.
    
    Lag features use past sales to predict current sales.
    - lag_1: yesterday's sales
    - lag_7: sales 7 days ago (weekly pattern)
    - lag_14: sales 14 days ago
    
    NO LEAKAGE: lag_N uses sales(t-N) to predict sales(t)
    """
    print("\nCreating lag features...")
    
    df = df.copy()
    df['group_key'] = df['item_id'] + '_' + df['store_id']
    
    # Lag 1 (yesterday)
    print("  - lag_1")
    df['lag_1'] = df.groupby('group_key', sort=False)['sales'].shift(1)
    
    # Lag 7 (same day last week)
    print("  - lag_7")
    df['lag_7'] = df.groupby('group_key', sort=False)['sales'].shift(7)
    
    # Lag 14 (same day 2 weeks ago)
    print("  - lag_14")
    df['lag_14'] = df.groupby('group_key', sort=False)['sales'].shift(14)
    
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    CRITICAL: All rolling features are SHIFTED by 1 to avoid leakage.
    We compute the rolling stat up to t-1, then use it to predict t.
    
    - rolling_mean_7: mean of last 7 days (shifted)
    - rolling_mean_14: mean of last 14 days (shifted)
    - rolling_std_7: std of last 7 days (shifted)
    """
    print("\nCreating rolling features (shifted to avoid leakage)...")
    
    # 7-day rolling mean (shifted by 1)
    print("  - rolling_mean_7")
    df['rolling_mean_7'] = (
        df.groupby('group_key', sort=False)['sales']
        .transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
    )
    
    # 14-day rolling mean (shifted by 1)
    print("  - rolling_mean_14")
    df['rolling_mean_14'] = (
        df.groupby('group_key', sort=False)['sales']
        .transform(lambda x: x.rolling(14, min_periods=1).mean().shift(1))
    )
    
    # 7-day rolling std (shifted by 1)
    print("  - rolling_std_7")
    df['rolling_std_7'] = (
        df.groupby('group_key', sort=False)['sales']
        .transform(lambda x: x.rolling(7, min_periods=2).std().shift(1))
    )
    
    return df


def create_calendar_features(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features.
    
    These features are deterministic and based on the date only.
    - day_of_week: 0=Monday, 6=Sunday
    - week_of_year: 1-52
    - is_weekend: 1 if Saturday or Sunday
    - is_event: 1 if there's an event/holiday
    """
    print("\nCreating calendar features...")
    
    # Day of week (0=Monday, 6=Sunday)
    print("  - day_of_week")
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Week of year
    print("  - week_of_year")
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Is weekend
    print("  - is_weekend")
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    
    # Month (useful for seasonality)
    print("  - month")
    df['month'] = df['date'].dt.month
    
    # Merge event data from calendar
    print("  - is_event")
    df = df.merge(calendar[['date', 'is_event']], on='date', how='left')
    df['is_event'] = df['is_event'].fillna(0).astype(int)
    
    return df


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final feature dataframe with proper column ordering.
    
    Output columns:
    - item_id, store_id, date: identifiers
    - target: sales (what we're predicting)
    - features: all engineered features
    """
    print("\nBuilding final feature dataframe...")
    
    # Rename sales to target
    df['target'] = df['sales']
    
    # Define feature columns
    lag_features = ['lag_1', 'lag_7', 'lag_14']
    rolling_features = ['rolling_mean_7', 'rolling_mean_14', 'rolling_std_7']
    calendar_features = ['day_of_week', 'week_of_year', 'is_weekend', 'month', 'is_event']
    
    all_features = lag_features + rolling_features + calendar_features
    
    # Select final columns
    output_cols = ['item_id', 'store_id', 'date', 'target'] + all_features
    df = df[output_cols].copy()
    
    return df, all_features


def validate_no_leakage(df: pd.DataFrame) -> None:
    """
    Validate that no leakage exists in the features.
    
    Check: for any row, features should not use information from that row's target.
    """
    print("\n--- Leakage Validation ---")
    
    # Get a single item-store group for validation
    first_item = df['item_id'].iloc[0]
    first_store = df['store_id'].iloc[0]
    sample = df[(df['item_id'] == first_item) & (df['store_id'] == first_store)].copy()
    sample = sample.sort_values('date').reset_index(drop=True)
    
    # Check 1: lag_1[i] should equal target[i-1]
    lag1_correct = (sample['lag_1'].iloc[1:].values == sample['target'].iloc[:-1].values).mean()
    print(f"✓ lag_1 correctly shifted: {lag1_correct*100:.1f}% match (should be 100%)")
    
    # Check 2: lag_7[i] should equal target[i-7]
    lag7_correct = (sample['lag_7'].iloc[7:].values == sample['target'].iloc[:-7].values).mean()
    print(f"✓ lag_7 correctly shifted: {lag7_correct*100:.1f}% match (should be 100%)")
    
    # Check 3: rolling_mean_7 should not include current day's sales
    # If correctly shifted, rolling_mean_7[7] = mean(target[0:7]) not including target[7]
    expected_roll = sample['target'].iloc[0:7].mean()
    actual_roll = sample['rolling_mean_7'].iloc[7]
    roll_correct = np.isclose(expected_roll, actual_roll)
    print(f"✓ rolling_mean_7 correctly shifted: {roll_correct} (expected {expected_roll:.2f}, got {actual_roll:.2f})")
    
    if lag1_correct > 0.99 and lag7_correct > 0.99 and roll_correct:
        print("✓ No leakage detected - all features correctly shifted!")
    else:
        print("⚠ WARNING: Possible leakage detected!")


def get_earliest_usable_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Find the earliest date where all features are valid (not NaN).
    
    Due to lag_14 and rolling features, we lose the first ~14 days.
    """
    # Count NaN per date
    feature_cols = [col for col in df.columns if col not in ['item_id', 'store_id', 'date', 'target']]
    
    # Find first date where we have complete features
    df_complete = df.dropna(subset=feature_cols)
    earliest = df_complete['date'].min()
    
    return earliest


def main():
    """Build feature pipeline."""
    project_dir = Path(__file__).parent.parent
    
    # Load data
    df = load_sales_data(project_dir)
    calendar = load_calendar_data(project_dir)
    
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique items: {df['item_id'].nunique()}, Unique stores: {df['store_id'].nunique()}")
    
    # Create features
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_calendar_features(df, calendar)
    
    # Build final dataframe
    df, feature_list = build_feature_dataframe(df)
    
    # Validate no leakage
    validate_no_leakage(df)
    
    # Get earliest usable date
    earliest_date = get_earliest_usable_date(df)
    print(f"\nEarliest usable date (after lags): {earliest_date.date()}")
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"\nFeatures created ({len(feature_list)}):")
    for f in feature_list:
        print(f"  - {f}")
    
    print(f"\nDataframe shape: {df.shape}")
    print(f"Total rows: {len(df):,}")
    print(f"Complete rows (no NaN): {len(df.dropna()):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Earliest usable date: {earliest_date.date()}")
    
    # Save to parquet
    output_path = project_dir / "features" / "features.parquet"
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
    
    # Also save feature list for reference
    feature_info = pd.DataFrame({
        'feature': feature_list,
        'type': ['lag', 'lag', 'lag', 'rolling', 'rolling', 'rolling', 
                 'calendar', 'calendar', 'calendar', 'calendar', 'calendar']
    })
    feature_info.to_csv(project_dir / "features" / "feature_list.csv", index=False)
    
    print("\n" + "="*60)
    print("Feature pipeline complete!")
    print("="*60)
    
    return df, feature_list


if __name__ == "__main__":
    df, features = main()
