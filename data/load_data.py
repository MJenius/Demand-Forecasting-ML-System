import pandas as pd
import os
from pathlib import Path

# Get the directory containing CSV files
data_dir = Path(__file__).parent / "raw" / "m5"

# Load all CSV files
csv_files = sorted(data_dir.glob('*.csv'))

if not csv_files:
    print("No CSV files found in the data directory.")
else:
    for csv_file in csv_files:
        try:
            # For large files, just read the shape without loading full data
            df = pd.read_csv(csv_file, nrows=1)
            # Get actual file dimensions more efficiently
            import csv as csv_module
            with open(csv_file) as f:
                rows = sum(1 for _ in f) - 1  # subtract 1 for header
            cols = len(df.columns)
            print(f"{csv_file.name:40} ({rows:>9}, {cols:>3})")
        except Exception as e:
            print(f"{csv_file.name:40} Error: {str(e)[:50]}")