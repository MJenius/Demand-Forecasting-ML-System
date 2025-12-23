"""
Prediction logger for drift detection monitoring
"""

import json
from datetime import datetime
from pathlib import Path
import logging


class PredictionLogger:
    """Log predictions for later drift detection and monitoring."""
    
    def __init__(self, log_dir: str = "inference/logs"):
        """Initialize logger with output directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard logging for warnings/errors
        self.logger = logging.getLogger(__name__)
        
    def log_prediction(self, item_id: str, store_id: str, date: str,
                      prediction: float, model_used: str, model_version: str,
                      features: dict = None, actual_value: float = None) -> None:
        """
        Log a prediction to JSON file for monitoring.
        
        Args:
            item_id: Item ID
            store_id: Store ID
            date: Prediction date
            prediction: Predicted value
            model_used: Which model made the prediction (lightgbm or moving_average)
            model_version: Model version (e.g., v1)
            features: Optional feature dict for debugging
            actual_value: Optional actual value for post-hoc evaluation
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "item_id": item_id,
            "store_id": store_id,
            "date": date,
            "prediction": float(prediction),
            "model_used": model_used,
            "model_version": model_version,
            "actual_value": float(actual_value) if actual_value is not None else None,
            "features": features
        }
        
        # Append to daily log file
        log_file = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write prediction log: {e}")
    
    def get_logs(self, date_str: str = None) -> list:
        """
        Retrieve logged predictions for a given date.
        
        Args:
            date_str: Date in YYYYMMDD format. If None, returns today's logs.
        
        Returns:
            List of log entries
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')
        
        log_file = self.log_dir / f"predictions_{date_str}.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        
        return logs
