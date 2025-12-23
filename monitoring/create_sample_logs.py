"""
Create more comprehensive sample prediction logs for drift detection testing.

This creates predictions with varying distributions to test drift detection
across multiple days.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_realistic_logs():
    """Create realistic prediction logs with drift for testing."""
    logs_dir = Path("inference/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample SKUs
    sample_skus = [
        ("FOODS_3_090", "CA_3"),
        ("FOODS_1_001", "CA_1"),
        ("HOUSEHOLD_1_116", "TX_1"),
        ("HOBBIES_1_001", "WI_1"),
        ("FOODS_2_050", "CA_2"),
        ("HOBBIES_2_100", "CA_3"),
        ("FOODS_3_200", "TX_1"),
        ("HOUSEHOLD_2_050", "WI_1"),
    ]
    
    print("Creating realistic prediction logs for drift testing...")
    
    # Generate predictions for 25 days (enough for reference + current windows)
    base_date = datetime.now() - timedelta(days=24)
    
    for day_offset in range(25):
        current_date = base_date + timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")
        log_file = logs_dir / f"predictions_{current_date.strftime('%Y%m%d')}.jsonl"
        
        # Create more predictions per day
        np.random.seed((day_offset * 1000 + len(sample_skus)) % (2**32 - 1))
        
        # Later days have higher/drifted predictions
        if day_offset < 14:
            # Reference period - base distribution
            base_mean = 10.0
            base_std = 3.0
        else:
            # Current period - slightly shifted distribution (subtle drift)
            base_mean = 12.5  # +25% mean shift
            base_std = 4.5    # +50% std shift
        
        with open(log_file, 'a') as f:
            for sku_idx, (item_id, store_id) in enumerate(sample_skus):
                # Create 4 predictions per SKU per day
                for pred_idx in range(4):
                    timestamp = current_date.replace(
                        hour=6 + pred_idx * 4,
                        minute=15 + (sku_idx * 3) % 60,
                        second=30
                    ).isoformat()
                    
                    # Alternate models
                    model_used = "moving_average" if (sku_idx + pred_idx) % 2 == 0 else "lightgbm"
                    
                    # Generate prediction from shifted distribution
                    prediction = max(0.0, np.random.normal(base_mean, base_std))
                    
                    log_entry = {
                        "timestamp": timestamp,
                        "item_id": item_id,
                        "store_id": store_id,
                        "date": date_str,
                        "prediction": round(prediction, 4),
                        "model_used": model_used,
                        "model_version": "v1",
                        "actual_value": None,
                        "features": {}
                    }
                    
                    f.write(json.dumps(log_entry) + '\n')
        
        print(f"  ✓ Day {day_offset+1:2d}: {date_str} ({8} SKUs × 4 preds = {len(sample_skus)*4} predictions)")
    
    print(f"\n✓ Created {25} days of prediction logs")
    print(f"  Total predictions: {25 * len(sample_skus) * 4}")
    print(f"  Drift introduced: Mean +25%, Std +50% from day 15 onwards\n")


if __name__ == "__main__":
    create_realistic_logs()
