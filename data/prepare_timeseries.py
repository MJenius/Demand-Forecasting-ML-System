"""
Reshape sales data from wide to long format.

Converts from:
    product_id | d_1 | d_2 | d_3 | ...

To:
    date | product_id | store_id | sales

This enables:
- Time-based splits
- Rolling windows
- Drift detection
- Online simulation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sales and calendar data."""
    logger.info("Loading sales_train_validation.csv...")
    sales_df = pd.read_csv(data_dir / "raw" / "m5" / "sales_train_validation.csv")
    
    logger.info("Loading calendar.csv...")
    calendar_df = pd.read_csv(data_dir / "raw" / "m5" / "calendar.csv")
    
    logger.info(f"Sales shape: {sales_df.shape}")
    logger.info(f"Calendar shape: {calendar_df.shape}")
    
    return sales_df, calendar_df


def reshape_sales(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape sales data from wide to long format.
    
    Args:
        sales_df: Wide format sales data with columns: id, item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, ...
        calendar_df: Calendar data with columns: date, d (where d is the day number)
    
    Returns:
        Long format dataframe with columns: date, item_id, store_id, sales
    """
    logger.info("Reshaping sales data...")
    
    # Get day columns (d_1, d_2, ..., d_n)
    day_cols = [col for col in sales_df.columns if col.startswith('d_')]
    logger.info(f"Found {len(day_cols)} day columns")
    
    # Keep only relevant ID columns
    id_cols = ['id', 'item_id', 'store_id']
    keep_cols = [col for col in id_cols if col in sales_df.columns] + day_cols
    sales_subset = sales_df[keep_cols].copy()
    
    # Melt from wide to long
    logger.info("Melting sales data...")
    sales_long = sales_subset.melt(
        id_vars=[col for col in id_cols if col in sales_subset.columns],
        value_vars=day_cols,
        var_name='d_col',
        value_name='sales'
    )
    
    # Extract day number from 'd_1' -> 1
    sales_long['d'] = sales_long['d_col'].str.replace('d_', '').astype(int)
    sales_long = sales_long.drop(columns=['d_col'])
    
    # Create a mapping from day number to date
    logger.info("Joining with calendar data...")
    calendar_df_copy = calendar_df.copy()
    
    # The calendar 'd' column might be string like 'd_1', extract the number
    if calendar_df_copy['d'].dtype == 'object':
        calendar_df_copy['d'] = calendar_df_copy['d'].str.replace('d_', '').astype(int)
    else:
        calendar_df_copy['d'] = calendar_df_copy['d'].astype(int)
    
    # Join with calendar to get actual dates
    sales_long = sales_long.merge(
        calendar_df_copy[['d', 'date']],
        on='d',
        how='left'
    )
    
    # Convert date to datetime
    sales_long['date'] = pd.to_datetime(sales_long['date'])
    
    # Select and reorder columns
    result = sales_long[['date', 'item_id', 'store_id', 'sales']].copy()
    
    # Sort by date, then item_id, then store_id
    result = result.sort_values(['date', 'item_id', 'store_id']).reset_index(drop=True)
    
    logger.info(f"Final shape: {result.shape}")
    logger.info(f"Date range: {result['date'].min()} to {result['date'].max()}")
    logger.info(f"Unique items: {result['item_id'].nunique()}")
    logger.info(f"Unique stores: {result['store_id'].nunique()}")
    
    return result


def main():
    """Main execution."""
    data_dir = Path(__file__).parent
    
    # Create output directory
    output_dir = data_dir / "processed"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    sales_df, calendar_df = load_data(data_dir)
    
    # Reshape
    sales_long = reshape_sales(sales_df, calendar_df)
    
    # Save to parquet
    output_path = output_dir / "sales_long.parquet"
    logger.info(f"Saving to {output_path}...")
    
    try:
        # Use more efficient parquet writing
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        logger.info("Converting to Arrow table...")
        table = pa.Table.from_pandas(sales_long)
        
        logger.info("Writing parquet file...")
        pq.write_table(table, str(output_path), compression='snappy')
    except Exception as e:
        logger.warning(f"PyArrow write failed, trying pandas: {e}")
        sales_long.to_parquet(output_path, index=False, compression='snappy')
    
    logger.info("Done!")
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Shape: {sales_long.shape}")
    print(f"  Columns: {sales_long.columns.tolist()}")
    print(f"  Date range: {sales_long['date'].min()} to {sales_long['date'].max()}")
    print(f"  Unique items: {sales_long['item_id'].nunique()}")
    print(f"  Unique stores: {sales_long['store_id'].nunique()}")
    print(f"  File: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
