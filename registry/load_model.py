"""
Utility to load the active model from the registry.

This module provides a simple interface to:
- Get the active model version
- Load the model, features, and SKU segments
- Apply the two-model strategy for predictions
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

REGISTRY_DIR = Path(__file__).parent
REGISTRY_FILE = REGISTRY_DIR / "registry.json"


def get_active_model_version() -> str:
    """Get the currently active model version."""
    with open(REGISTRY_FILE, 'r') as f:
        registry = json.load(f)
    return registry['active_version']


def load_active_model() -> dict:
    """Load the active model from registry."""
    active_version = get_active_model_version()
    model_dir = REGISTRY_DIR / "models" / active_version
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"Loading active model: {active_version}")
    
    # Load model
    model_file = model_dir / "model.joblib"
    if not model_file.exists():
        model_file = model_dir / "lgb_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model binary not found in {model_dir}")
    model = joblib.load(model_file)
    print(f"  [OK] Model loaded from {model_file.name}")

    
    # Load metadata
    with open(model_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    print(f"  [OK] Metadata loaded")
    
    # Load features
    with open(model_dir / "features.json", 'r') as f:
        features = json.load(f)
    print(f"  [OK] Features loaded ({len(features)} features)")
    
    # Load SKU segments
    segments = pd.read_csv(model_dir / "sku_segments.csv")
    print(f"  [OK] SKU segments loaded ({len(segments):,} SKUs)")
    
    # Load metrics
    with open(model_dir / "metrics.json", 'r') as f:
        metrics = json.load(f)
    print(f"  [OK] Metrics loaded")

    # Load lineage manifest if present
    lineage_id = metadata.get("lineage_id")
    lineage_file = model_dir / "lineage.json"
    if lineage_file.exists():
        try:
            lineage_data = json.loads(lineage_file.read_text(encoding="utf-8"))
            lineage_id = lineage_data.get("lineage_id", lineage_id)
        except Exception:
            pass
    if not lineage_id:
        lineage_id = "unknown"
    metadata["lineage_id"] = lineage_id
    
    return {
        'version': active_version,
        'model': model,
        'metadata': metadata,
        'features': features,
        'segments': segments,
        'metrics': metrics,
        'lineage_id': lineage_id,
    }


def make_prediction(X: pd.DataFrame, item_ids: np.ndarray, store_ids: np.ndarray,
                   model_bundle: dict) -> np.ndarray:
    """
    Make predictions using the two-model strategy.
    
    Args:
        X: Feature dataframe
        item_ids: Item IDs
        store_ids: Store IDs
        model_bundle: Loaded model bundle from load_active_model()
    
    Returns:
        Predictions array
    """
    segments = model_bundle['segments']
    model = model_bundle['model']
    features = model_bundle['features']
    
    # Create segment lookup
    segments['key'] = segments['item_id'] + '_' + segments['store_id']
    normal_volume_keys = segments[segments['mean_sales'] >= 1.5]['key'].set_index('key').index
    
    # Initialize predictions
    predictions = np.zeros(len(X))
    
    # Split predictions
    for i, (idx, row) in enumerate(X.iterrows()):
        sku_key = f"{item_ids[i]}_{store_ids[i]}"
        
        if sku_key in normal_volume_keys.tolist():
            # Use LightGBM for normal-volume
            pred = model.predict(row[features].values.reshape(1, -1))[0]
        else:
            # Use 7-Day MA for low-volume
            # Note: This is a placeholder - in production, you'd load the historical data
            pred = row.get('rolling_mean_7', 0.0)
        
        predictions[i] = max(0, pred)  # Clip to non-negative
    
    return predictions


def print_registry_info() -> None:
    """Print information about the model registry."""
    with open(REGISTRY_FILE, 'r') as f:
        registry = json.load(f)
    
    active = registry['active_version']
    model_info = registry['models'][active]
    
    print("\n" + "=" * 60)
    print("MODEL REGISTRY INFO")
    print("=" * 60)
    print(f"\nActive Model: {active}")
    print(f"Status: {model_info['status']}")
    print(f"Registered: {model_info['registered_at']}")
    
    print(f"\nMetrics:")
    print(f"  MAE:  {model_info['metrics']['MAE']:.4f}")
    print(f"  RMSE: {model_info['metrics']['RMSE']:.4f}")
    print(f"  MAPE: {model_info['metrics']['MAPE']:.2f}%")
    
    print(f"\nImprovement vs Baseline (7-Day MA):")
    comparison = model_info['comparison_to_baseline']
    print(f"  MAE:  +{comparison['MAE']['improvement_pct']:.2f}%")
    print(f"  RMSE: +{comparison['RMSE']['improvement_pct']:.2f}%")
    print(f"  MAPE: +{comparison['MAPE']['improvement_pct']:.2f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_registry_info()
    print("\nLoading model...")
    model_bundle = load_active_model()
    print(f"\n✅ Model {model_bundle['version']} ready for inference")
