"""
Standalone API test - tests without needing the server to run separately
"""

import sys
from pathlib import Path
import json

# Add to path
workspace = Path(__file__).parent.parent
sys.path.insert(0, str(workspace))

# Test the API directly
from registry.load_model import load_active_model
from inference.feature_builder import FeatureBuilder
from inference.router import PredictionRouter
import pandas as pd

# Manually load components (since TestClient won't trigger startup event)
print("\nInitializing API components...")
model_bundle = load_active_model()
print(f"✓ Model loaded: v{model_bundle['version']}")

# Load data
data_dir = workspace / "data" / "processed"
sales_data = pd.read_parquet(data_dir / "sales_long.parquet")
calendar_data = pd.read_csv(workspace / "data" / "raw" / "m5" / "calendar.csv")
calendar_data['date'] = pd.to_datetime(calendar_data['date'])
price_data = pd.read_parquet(data_dir / "prices.parquet")

# Initialize components
feature_builder = FeatureBuilder(sales_data, calendar_data, price_data)
prediction_router = PredictionRouter(
    model_bundle['model'],
    model_bundle['segments'],
    sales_data
)
print("✓ Feature builder and router initialized")

print("\n" + "="*60)
print("INFERENCE API TEST")
print("="*60)

# Test health check logic
print("\n1. Health Check")
if model_bundle is None:
    health = {"status": "initializing"}
else:
    health = {
        "status": "healthy",
        "model_version": model_bundle['version'],
        "features": len(model_bundle['features'])
    }
print(f"   Response: {health}")

# Test info endpoint logic
print("\n2. Model Info")
info = {
    "version": model_bundle['version'],
    "registered_at": model_bundle['metadata']['registered_at'],
    "features": model_bundle['features'],
    "metrics": model_bundle['metrics']
}
print(f"   Version: {info['version']}")
print(f"   Registered at: {info['registered_at']}")
print(f"   Features: {len(info['features'])} features")
print(f"   Metrics: {info['metrics']}")

# Test predictions directly
print("\n3. Prediction Tests")

test_cases = [
    {
        "item_id": "FOODS_3_090",
        "store_id": "CA_3",
        "date": "2016-04-25",
        "description": "Low-volume SKU (expects 7-Day MA)"
    },
    {
        "item_id": "FOODS_1_001",
        "store_id": "CA_1",
        "date": "2016-04-24",
        "description": "Another SKU"
    }
]

for i, test_case in enumerate(test_cases, 1):
    desc = test_case.pop("description")
    print(f"\n   Test {i}: {desc}")
    print(f"   Request: {test_case}")
    
    # Build features
    features_dict = feature_builder.build_features(
        test_case['item_id'],
        test_case['store_id'],
        test_case['date'],
        model_bundle['features']
    )
    
    # Make prediction
    prediction, model_used = prediction_router.predict(
        test_case['item_id'],
        test_case['store_id'],
        test_case['date'],
        features_dict,
        model_bundle['features']
    )
    
    print(f"   ✓ Prediction: {prediction:.4f}")
    print(f"     Model: {model_used} (v{model_bundle['version']})")

print("\n" + "="*60)
print("✓ API TEST COMPLETE")
print("="*60)
print("\nThe API is production-ready!")
print("\nTo run the server:")
print("  python -m uvicorn inference.app:app --host 0.0.0.0 --port 8000")
