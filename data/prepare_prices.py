"""
Prepare price data for inference - converts sell_prices.csv to parquet format for fast loading
"""

import pandas as pd
from pathlib import Path
import sys

# Setup paths
data_dir = Path(__file__).parent
raw_dir = data_dir / "raw" / "m5"
processed_dir = data_dir / "processed"

processed_dir.mkdir(parents=True, exist_ok=True)

# Load calendar to get date mapping
print("Loading calendar...")
calendar = pd.read_csv(raw_dir / "calendar.csv")
calendar['date'] = pd.to_datetime(calendar['date'])
calendar = calendar[['d', 'date']].rename(columns={'d': 'd_number'})

# Load sell prices
print("Loading sell prices...")
prices = pd.read_csv(raw_dir / "sell_prices.csv")

# Convert wm_yr_wk and item_id columns to match format
print("Processing price data...")

# Melt prices from wide to long format (if needed)
# The sell_prices.csv is already in (store_id, item_id, wm_yr_wk, sell_price) format
# We need to convert wm_yr_wk to dates

# Load calendar with wm_yr_wk mapping
calendar_full = pd.read_csv(raw_dir / "calendar.csv")
calendar_full['date'] = pd.to_datetime(calendar_full['date'])

# Create wm_yr_wk to date mapping (use first day of that week)
wm_yr_wk_to_date = calendar_full.groupby('wm_yr_wk')['date'].min().reset_index()
wm_yr_wk_to_date.columns = ['wm_yr_wk', 'date']

# Merge to get dates
prices = prices.merge(wm_yr_wk_to_date, on='wm_yr_wk', how='left')

# Keep only needed columns
prices = prices[['item_id', 'store_id', 'date', 'sell_price']].copy()

# Sort for efficient retrieval
prices = prices.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

print(f"Processed {len(prices):,} price records")
print(f"Date range: {prices['date'].min()} to {prices['date'].max()}")
print(f"Unique items: {prices['item_id'].nunique()}")
print(f"Unique stores: {prices['store_id'].nunique()}")

# Save to parquet
output_path = processed_dir / "prices.parquet"
prices.to_parquet(output_path, engine='pyarrow', index=False)
print(f"\n✓ Saved to {output_path}")
