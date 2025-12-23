"""
Test script for inference API
"""

import sys
from pathlib import Path
import pandas as pd

# Add to path
workspace = Path(__file__).parent.parent
sys.path.insert(0, str(workspace))

from registry.load_model import load_active_model
from inference.feature_builder import FeatureBuilder
from inference.router import PredictionRouter

# Load model
print("Loading model...")
model_bundle = load_active_model()
print(f"✓ Model version: {model_bundle['version']}")
print(f"✓ Features: {model_bundle['features']}")
print(f"✓ Segments: {model_bundle['segments'].shape[0]} SKUs")

# Load data
print("\nLoading data for feature engineering...")
data_dir = workspace / "data" / "processed"

sales_data = pd.read_parquet(data_dir / "sales_long.parquet")
print(f"✓ Sales data: {len(sales_data)} rows")

calendar_data = pd.read_csv(workspace / "data" / "raw" / "m5" / "calendar.csv")
calendar_data['date'] = pd.to_datetime(calendar_data['date'])
print(f"✓ Calendar data: {len(calendar_data)} rows")

price_data = pd.read_parquet(data_dir / "prices.parquet")
print(f"✓ Price data: {len(price_data)} rows")

# Initialize components
print("\nInitializing feature builder and router...")
feature_builder = FeatureBuilder(sales_data, calendar_data, price_data)
prediction_router = PredictionRouter(
    model_bundle['model'],
    model_bundle['segments'],
    sales_data
)

# Test prediction
print("\n" + "="*60)
print("TEST: Single SKU Prediction")
print("="*60)

# Use a known SKU from the validation set
item_id = "FOODS_3_090"
store_id = "CA_3"
target_date = "2016-04-25"

print(f"\nPredicting for {item_id} in {store_id} on {target_date}")

# Check segment
segment = prediction_router.get_segment(item_id, store_id)
print(f"Segment: {segment}")

# Build features
try:
    features_dict = feature_builder.build_features(
        item_id,
        store_id,
        target_date,
        model_bundle['features']
    )
    print(f"✓ Features built successfully ({len(features_dict)} features)")
    
    # Show sample features
    sample_features = {k: v for i, (k, v) in enumerate(features_dict.items()) if i < 5}
    print(f"Sample features: {sample_features}")
    
    # Make prediction
    prediction, model_used = prediction_router.predict(
        item_id,
        store_id,
        target_date,
        features_dict,
        model_bundle['features']
    )
    
    print(f"\n✓ Prediction successful!")
    print(f"  - Value: {prediction:.4f}")
    print(f"  - Model used: {model_used}")
    print(f"  - Model version: {model_bundle['version']}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("API READY FOR FASTAPI TEST")
print("="*60)
print("\nRun: python -m uvicorn inference.app:app --reload")
print("Test endpoint: POST http://localhost:8000/predict")
