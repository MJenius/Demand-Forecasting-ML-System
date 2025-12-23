"""
Step 5: Model Registry & Deployment Readiness

This script registers trained models in a versioned registry with:
- Explicit versioning (v1, v2, etc.)
- Complete metadata (training dates, thresholds, features)
- Metrics for model comparison
- Promotion logic (only replace current if beats on MAE & RMSE)
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import subprocess

BASE_DIR = Path(__file__).parent.parent
REGISTRY_DIR = BASE_DIR / "registry"
MODELS_DIR = REGISTRY_DIR / "models"
SEGMENTED_MODEL_DIR = REGISTRY_DIR / "lightgbm_segmented"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"


def get_git_commit_hash() -> str:
    """Get current git commit hash for tracking."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception as e:
        print(f"Warning: Could not get git commit hash: {e}")
        return "unknown"


def load_segmented_model_artifacts() -> dict:
    """Load the trained segmented model and its metadata."""
    print("Loading segmented model artifacts...")
    
    # Load model
    model_path = SEGMENTED_MODEL_DIR / "model.joblib"
    model = joblib.load(model_path)
    print(f"  ✓ Model loaded from {model_path}")
    
    # Load metadata
    metadata_path = SEGMENTED_MODEL_DIR / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"  ✓ Metadata loaded")
    
    # Load metrics
    metrics_path = SEGMENTED_MODEL_DIR / "metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    print(f"  ✓ Metrics loaded")
    
    # Load SKU segments
    segments_path = SEGMENTED_MODEL_DIR / "sku_segments.csv"
    segments = pd.read_csv(segments_path)
    print(f"  ✓ SKU segments loaded ({len(segments):,} total)")
    
    # Load features
    features_path = SEGMENTED_MODEL_DIR / "features.json"
    with open(features_path, 'r') as f:
        features = json.load(f)
    print(f"  ✓ Features loaded ({len(features)} features)")
    
    return {
        'model': model,
        'metadata': metadata,
        'metrics': metrics,
        'segments': segments,
        'features': features
    }


def create_model_version_dir(version: str) -> Path:
    """Create versioned model directory structure."""
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created model directory: {version_dir}")
    return version_dir


def save_model_artifacts(version: str, artifacts: dict) -> Path:
    """Save model artifacts to versioned registry."""
    print(f"\n--- Saving Model {version} Artifacts ---")
    
    version_dir = create_model_version_dir(version)
    
    # 1. Save LightGBM model
    model_path = version_dir / "lgb_model.pkl"
    joblib.dump(artifacts['model'], model_path)
    print(f"  ✓ Model saved to {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    
    # 2. Save comprehensive metadata
    metadata = {
        'model_version': version,
        'model_type': 'LightGBM + 7-Day MA (Segmented)',
        'registered_at': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash(),
        'strategy': {
            'name': 'Two-Model Strategy',
            'description': 'LightGBM for normal-volume SKUs, 7-Day MA for low-volume',
            'volume_threshold': 1.5,
            'normal_volume_count': len(artifacts['segments'][artifacts['segments']['mean_sales'] >= 1.5]),
            'low_volume_count': len(artifacts['segments'][artifacts['segments']['mean_sales'] < 1.5])
        },
        'training': {
            'end_date': '2016-03-27',
            'validation_start': '2016-03-28',
            'validation_end': '2016-04-24',
            'training_rows': 1685190,
            'validation_rows': 140000
        },
        'features': {
            'count': len(artifacts['features']),
            'list': artifacts['features'],
            'lags': ['lag_1', 'lag_7', 'lag_14', 'lag_21', 'lag_28'],
            'rolling': ['rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28', 
                       'rolling_std_7', 'rolling_std_14', 'rolling_std_28',
                       'rolling_max_7', 'rolling_min_7'],
            'price': ['sell_price', 'price_norm', 'price_change', 'is_price_reduced'],
            'calendar': ['day_of_week', 'week_of_year', 'is_weekend', 'month',
                        'day_of_month', 'is_month_end', 'is_month_start', 'is_event']
        },
        'lightgbm_params': {
            'objective': 'regression',
            'num_leaves': 127,
            'max_depth': 10,
            'learning_rate': 0.03,
            'best_iteration': artifacts['metadata'].get('best_iteration', 'unknown')
        }
    }
    
    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to {metadata_path}")
    
    # 3. Save metrics
    metrics = {
        'model_metrics': artifacts['metrics']['model_metrics'],
        'baseline_metrics': artifacts['metrics']['baseline_metrics'],
        'comparison': artifacts['metrics']['comparison']
    }
    
    metrics_path = version_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to {metrics_path}")
    
    # 4. Save SKU segments
    segments_path = version_dir / "sku_segments.csv"
    artifacts['segments'].to_csv(segments_path, index=False)
    print(f"  ✓ SKU segments saved to {segments_path}")
    
    # 5. Save feature list
    features_path = version_dir / "features.json"
    with open(features_path, 'w') as f:
        json.dump(artifacts['features'], f, indent=2)
    print(f"  ✓ Features saved to {features_path}")
    
    return version_dir


def load_registry() -> dict:
    """Load the central registry file."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    else:
        return {
            'active_version': None,
            'models': {},
            'promotion_history': []
        }


def should_promote_model(new_version: str, current_version: str, 
                         new_metrics: dict, current_metrics: dict) -> tuple:
    """
    Determine if new model should be promoted.
    
    Promotion rule: New model replaces current only if:
    - MAE improves (lower is better)
    - RMSE improves (lower is better)
    
    Returns (should_promote, reason)
    """
    if current_version is None:
        return True, "No current model - first model"
    
    new_mae = new_metrics['model_metrics']['MAE']
    new_rmse = new_metrics['model_metrics']['RMSE']
    
    current_mae = current_metrics['model_metrics']['MAE']
    current_rmse = current_metrics['model_metrics']['RMSE']
    
    mae_improves = new_mae < current_mae
    rmse_improves = new_rmse < current_rmse
    
    if mae_improves and rmse_improves:
        mae_pct = ((current_mae - new_mae) / current_mae) * 100
        rmse_pct = ((current_rmse - new_rmse) / current_rmse) * 100
        reason = f"MAE improves {mae_pct:.2f}%, RMSE improves {rmse_pct:.2f}%"
        return True, reason
    else:
        reason = f"MAE: {('improves' if mae_improves else 'degrades')}, RMSE: {('improves' if rmse_improves else 'degrades')}"
        return False, reason


def register_model(version: str, artifacts: dict) -> dict:
    """
    Register model in the central registry.
    
    Updates registry.json with model info and applies promotion logic.
    """
    print(f"\n--- Registering Model {version} ---")
    
    # Load current registry
    registry = load_registry()
    current_version = registry.get('active_version')
    
    # Save model artifacts
    version_dir = save_model_artifacts(version, artifacts)
    
    # Get model metrics for promotion decision
    new_metrics = artifacts['metrics']
    
    # Check if should promote
    should_promote = False
    reason = "Not evaluated"
    
    if current_version is None:
        should_promote = True
        reason = "First model"
    else:
        # Load current model metrics for comparison
        current_metrics_path = MODELS_DIR / current_version / "metrics.json"
        if current_metrics_path.exists():
            with open(current_metrics_path, 'r') as f:
                current_metrics = json.load(f)
            should_promote, reason = should_promote_model(
                version, current_version, new_metrics, current_metrics
            )
    
    # Update registry
    registry['models'][version] = {
        'registered_at': datetime.now().isoformat(),
        'status': 'active' if should_promote else 'inactive',
        'metrics': new_metrics['model_metrics'],
        'comparison_to_baseline': new_metrics['comparison'],
        'path': str(version_dir)
    }
    
    if should_promote:
        old_version = registry.get('active_version')
        registry['active_version'] = version
        registry['promotion_history'].append({
            'timestamp': datetime.now().isoformat(),
            'from_version': old_version,
            'to_version': version,
            'reason': reason
        })
    
    # Save updated registry
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"  ✓ Registry updated")
    print(f"\nPromotion Decision:")
    print(f"  Status: {'PROMOTED' if should_promote else 'NOT PROMOTED'}")
    print(f"  Reason: {reason}")
    if should_promote:
        print(f"  Active model: {version}")
    
    return registry


def print_registry_status() -> None:
    """Print current registry status."""
    print("\n" + "=" * 60)
    print("MODEL REGISTRY STATUS")
    print("=" * 60)
    
    registry = load_registry()
    
    if not registry['models']:
        print("No models registered yet")
        return
    
    print(f"\nActive Model: {registry['active_version']}")
    print(f"Total Registered: {len(registry['models'])} models")
    
    print(f"\nModel Inventory:")
    for version, info in registry['models'].items():
        status = "✅ ACTIVE" if version == registry['active_version'] else "❌ inactive"
        metrics = info['metrics']
        print(f"\n  {version} [{status}]")
        print(f"    MAE:  {metrics['MAE']:.4f}")
        print(f"    RMSE: {metrics['RMSE']:.4f}")
        print(f"    MAPE: {metrics['MAPE']:.2f}%")
        print(f"    Registered: {info['registered_at']}")
    
    if registry['promotion_history']:
        print(f"\nPromotion History:")
        for event in registry['promotion_history']:
            from_v = event['from_version'] or 'none'
            print(f"  {event['timestamp']}: {from_v} → {event['to_version']}")
            print(f"    Reason: {event['reason']}")
    
    print("\n" + "=" * 60)


def main():
    """Register v1 model in the registry."""
    print("=" * 60)
    print("STEP 5A: MODEL REGISTRY & DEPLOYMENT READINESS")
    print("=" * 60)
    
    # Load trained model artifacts
    artifacts = load_segmented_model_artifacts()
    
    # Register v1
    version = "v1"  # No prefix here - already named v1
    registry = register_model(version, artifacts)
    
    # Print status
    print_registry_status()
    
    print("\n✅ Model Registry Ready for Deployment")
    print(f"\nRegistry Location: {REGISTRY_FILE}")
    print(f"Model Directory: {MODELS_DIR / version}")
    
    return registry


if __name__ == "__main__":
    main()
