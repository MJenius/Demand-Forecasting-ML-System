"""
Test prediction collection and demonstrate logging workflow.

This script:
1. Creates sample prediction logs (simulating API predictions)
2. Ingests them into the SQLite database
3. Shows example queries and statistics
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.collect_predictions import PredictionCollector


def create_sample_logs():
    """Create sample prediction logs for testing."""
    logs_dir = Path("inference/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample predictions for the last 3 days
    today = datetime.now()
    sample_skus = [
        ("FOODS_3_090", "CA_3"),
        ("FOODS_1_001", "CA_1"),
        ("HOUSEHOLD_1_116", "TX_1"),
        ("HOBBIES_1_001", "WI_1"),
        ("FOODS_2_050", "CA_2"),
    ]
    
    print("Creating sample prediction logs...")
    
    for day_offset in range(3):
        current_date = today - timedelta(days=day_offset)
        date_str = current_date.strftime("%Y-%m-%d")
        log_file = logs_dir / f"predictions_{current_date.strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            for sku_idx, (item_id, store_id) in enumerate(sample_skus):
                # Create 2-3 predictions per SKU per day
                for pred_idx in range(2):
                    timestamp = current_date.replace(
                        hour=8 + pred_idx,
                        minute=15 + sku_idx * 10,
                        second=30
                    ).isoformat()
                    
                    # Alternate between models
                    model_used = "moving_average" if (sku_idx + pred_idx) % 2 == 0 else "lightgbm"
                    
                    # Generate realistic-looking prediction
                    import random
                    random.seed(hash((item_id, store_id, date_str, pred_idx)))
                    prediction = round(random.uniform(0.5, 50.0), 4)
                    
                    log_entry = {
                        "timestamp": timestamp,
                        "item_id": item_id,
                        "store_id": store_id,
                        "date": date_str,
                        "prediction": prediction,
                        "model_used": model_used,
                        "model_version": "v1",
                        "actual_value": None,
                        "features": {}  # Omitted for brevity
                    }
                    
                    f.write(json.dumps(log_entry) + '\n')
        
        print(f"  ✓ Created {log_file.name}")
    
    print(f"✓ Sample logs created in {logs_dir}\n")


def main():
    """Main workflow demonstration."""
    print("="*60)
    print("PREDICTION COLLECTION - STEP 7 TEST")
    print("="*60)
    
    # Step 1: Create sample logs
    create_sample_logs()
    
    # Step 2: Initialize collector
    print("Initializing PredictionCollector...")
    collector = PredictionCollector(db_path="monitoring/predictions.db")
    print("✓ Database initialized\n")
    
    # Step 3: Ingest logs
    print("Ingesting prediction logs from JSONL files...")
    ingested_count = collector.ingest_logs(logs_dir="inference/logs")
    print(f"✓ Ingested {ingested_count} prediction records\n")
    
    # Step 4: Get database statistics
    print("="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    stats = collector.get_database_stats()
    
    print(f"\nOverall Stats:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Unique dates: {stats['unique_dates']}")
    print(f"  Unique items: {stats['unique_items']}")
    print(f"  Unique stores: {stats['unique_stores']}")
    
    if stats['date_range']['start']:
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    
    print(f"\nBy Model Version:")
    for version, count in stats['by_model_version'].items():
        print(f"  {version}: {count} predictions")
    
    print(f"\nBy Model Type:")
    for model_type, count in stats['by_model_type'].items():
        print(f"  {model_type}: {count} predictions")
    
    # Step 5: Get recent predictions
    print("\n" + "="*60)
    print("RECENT PREDICTIONS (Last 7 days)")
    print("="*60 + "\n")
    
    recent = collector.get_recent_predictions(days=7)
    if len(recent) > 0:
        # Show first 5
        print("Sample of first 5 predictions:")
        print(recent.head(5).to_string(index=False))
        
        # Show statistics by model version
        print("\n" + "="*60)
        print("PREDICTION STATISTICS BY MODEL VERSION")
        print("="*60 + "\n")
        
        pred_stats = collector.get_prediction_stats(days=7)
        for stats_entry in pred_stats:
            print(f"Model Version: {stats_entry['model_version']}")
            print(f"  Count: {stats_entry['count']}")
            print(f"  Mean prediction: {stats_entry['mean']:.4f}")
            print(f"  Min: {stats_entry['min']:.4f}, Max: {stats_entry['max']:.4f}")
            print(f"  Unique items: {stats_entry['unique_items']}")
            print(f"  Unique stores: {stats_entry['unique_stores']}")
            print()
    else:
        print("No recent predictions found")
    
    # Step 6: Demonstrate export to Parquet
    print("="*60)
    print("EXPORTING TO PARQUET")
    print("="*60 + "\n")
    
    parquet_path = "monitoring/predictions_export.parquet"
    collector.export_to_parquet(parquet_path)
    print(f"✓ Exported predictions to {parquet_path}\n")
    
    # Step 7: Show database location
    print("="*60)
    print("NEXT STEPS")
    print("="*60 + "\n")
    print(f"Database location: {collector.db_path}")
    print(f"Logs directory: inference/logs/")
    print(f"\nTo ingest new logs: collector.ingest_logs()")
    print("To query: collector.get_recent_predictions(), collector.get_prediction_stats()")
    print("\nReady for drift detection in Step 7B!")


if __name__ == "__main__":
    main()
