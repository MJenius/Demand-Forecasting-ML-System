"""
Collect and persist prediction logs for monitoring.

This module:
1. Reads prediction logs from inference/logs/ (JSONL format)
2. Persists to SQLite database for efficient querying
3. Provides interface to access predictions for drift detection

The SQLite database becomes the single source of truth for all predictions
and enables efficient time-series analysis for drift detection.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PredictionCollector:
    """Collect and manage prediction logs in SQLite database."""
    
    def __init__(self, db_path: str = "monitoring/predictions.db"):
        """
        Initialize the prediction collector.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema if not exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                item_id TEXT NOT NULL,
                store_id TEXT NOT NULL,
                date TEXT NOT NULL,
                prediction REAL NOT NULL,
                model_used TEXT NOT NULL,
                model_version TEXT NOT NULL,
                actual_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(item_id, store_id, date, model_version)
            )
        """)
        
        # Create index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_date 
            ON predictions(date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_item_store 
            ON predictions(item_id, store_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_version 
            ON predictions(model_version)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON predictions(timestamp)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def add_prediction(self, timestamp: str, item_id: str, store_id: str, date: str,
                      prediction: float, model_used: str, model_version: str,
                      actual_value: float = None) -> None:
        """
        Add a single prediction to the database.
        
        Args:
            timestamp: ISO format timestamp (e.g., "2025-12-23T17:25:34.123456")
            item_id: Item ID
            store_id: Store ID
            date: Prediction date (YYYY-MM-DD)
            prediction: Predicted value
            model_used: Model name (lightgbm or moving_average)
            model_version: Model version (e.g., v1)
            actual_value: Optional actual sales (for post-hoc evaluation)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO predictions 
                (timestamp, item_id, store_id, date, prediction, model_used, model_version, actual_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, item_id, store_id, date, prediction, model_used, model_version, actual_value))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error adding prediction: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def ingest_logs(self, logs_dir: str = "inference/logs") -> int:
        """
        Ingest all prediction logs from JSONL files into database.
        
        Reads files like predictions_YYYYMMDD.jsonl and persists to SQLite.
        
        Args:
            logs_dir: Directory containing JSONL log files
        
        Returns:
            Number of predictions ingested
        """
        logs_dir = Path(logs_dir)
        if not logs_dir.exists():
            logger.warning(f"Logs directory not found: {logs_dir}")
            return 0
        
        total_ingested = 0
        
        # Process all JSONL files in the directory
        for log_file in sorted(logs_dir.glob("predictions_*.jsonl")):
            logger.info(f"Processing {log_file.name}...")
            
            with open(log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line)
                        
                        # Extract fields
                        self.add_prediction(
                            timestamp=entry['timestamp'],
                            item_id=entry['item_id'],
                            store_id=entry['store_id'],
                            date=entry['date'],
                            prediction=entry['prediction'],
                            model_used=entry['model_used'],
                            model_version=entry['model_version'],
                            actual_value=entry.get('actual_value')
                        )
                        
                        total_ingested += 1
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"{log_file.name}:{line_num} - JSON decode error: {e}")
                    except KeyError as e:
                        logger.warning(f"{log_file.name}:{line_num} - Missing field: {e}")
                    except Exception as e:
                        logger.warning(f"{log_file.name}:{line_num} - Error: {e}")
        
        logger.info(f"Ingested {total_ingested} predictions from JSONL logs")
        return total_ingested
    
    def get_predictions_for_date(self, date: str) -> pd.DataFrame:
        """
        Get all predictions for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
        
        Returns:
            DataFrame with predictions
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, item_id, store_id, date, prediction, model_used, 
                   model_version, actual_value
            FROM predictions
            WHERE date = ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()
        
        return df
    
    def get_predictions_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get all predictions within a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with predictions
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, item_id, store_id, date, prediction, model_used, 
                   model_version, actual_value
            FROM predictions
            WHERE date >= ? AND date <= ?
            ORDER BY date, timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        return df
    
    def get_predictions_by_model_version(self, model_version: str) -> pd.DataFrame:
        """
        Get all predictions from a specific model version.
        
        Args:
            model_version: Model version (e.g., v1)
        
        Returns:
            DataFrame with predictions
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, item_id, store_id, date, prediction, model_used, 
                   model_version, actual_value
            FROM predictions
            WHERE model_version = ?
            ORDER BY date, timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(model_version,))
        conn.close()
        
        return df
    
    def get_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """
        Get predictions from the last N days.
        
        Args:
            days: Number of days to look back
        
        Returns:
            DataFrame with recent predictions
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, item_id, store_id, date, prediction, model_used, 
                   model_version, actual_value
            FROM predictions
            WHERE date >= date((SELECT MAX(date) FROM predictions), '-' || ? || ' days')
            ORDER BY date, timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(days,))
        conn.close()
        
        return df
    
    def get_prediction_stats(self, date: str = None, days: int = None) -> dict:
        """
        Get statistics about predictions.
        
        Args:
            date: Specific date (YYYY-MM-DD), if None uses recent days
            days: Number of recent days to analyze (default 7 if date is None)
        
        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        
        if date:
            query = """
                SELECT 
                    COUNT(*) as count,
                    AVG(prediction) as mean,
                    MIN(prediction) as min,
                    MAX(prediction) as max,
                    COUNT(DISTINCT item_id) as unique_items,
                    COUNT(DISTINCT store_id) as unique_stores,
                    model_version
                FROM predictions
                WHERE date = ?
                GROUP BY model_version
            """
            result = pd.read_sql_query(query, conn, params=(date,))
        else:
            days = days or 7
            query = """
                SELECT 
                    COUNT(*) as count,
                    AVG(prediction) as mean,
                    MIN(prediction) as min,
                    MAX(prediction) as max,
                    COUNT(DISTINCT item_id) as unique_items,
                    COUNT(DISTINCT store_id) as unique_stores,
                    COUNT(DISTINCT date) as unique_dates,
                    model_version
                FROM predictions
                WHERE date >= date((SELECT MAX(date) FROM predictions), '-' || ? || ' days')
                GROUP BY model_version
            """
            result = pd.read_sql_query(query, conn, params=(days,))
        
        conn.close()
        
        return result.to_dict('records') if len(result) > 0 else {}
    
    def update_actual_value(self, item_id: str, store_id: str, date: str, 
                           actual_value: float) -> None:
        """
        Update actual sales value for post-hoc evaluation.
        
        This is called once ground truth becomes available (e.g., end of day).
        
        Args:
            item_id: Item ID
            store_id: Store ID
            date: Prediction date
            actual_value: Actual sales value
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE predictions
                SET actual_value = ?
                WHERE item_id = ? AND store_id = ? AND date = ?
            """, (actual_value, item_id, store_id, date))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating actual value: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def export_to_parquet(self, output_path: str) -> None:
        """
        Export all predictions to Parquet format.
        
        Useful for integration with data warehouses or analytics tools.
        
        Args:
            output_path: Path to output Parquet file
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, item_id, store_id, date, prediction, model_used, 
                   model_version, actual_value, created_at
            FROM predictions
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert to proper types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = pd.to_datetime(df['date'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Save to Parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(df)} predictions to {output_path}")
    
    def get_database_stats(self) -> dict:
        """
        Get overall database statistics.
        
        Returns:
            Dictionary with counts and date ranges
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT date) FROM predictions")
        unique_dates = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT item_id) FROM predictions")
        unique_items = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT store_id) FROM predictions")
        unique_stores = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date), MAX(date) FROM predictions")
        date_range = cursor.fetchone()
        
        cursor.execute("""
            SELECT model_version, COUNT(*) 
            FROM predictions 
            GROUP BY model_version
        """)
        model_versions = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT model_used, COUNT(*) 
            FROM predictions 
            GROUP BY model_used
        """)
        model_types = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_predictions': total_count,
            'unique_dates': unique_dates,
            'unique_items': unique_items,
            'unique_stores': unique_stores,
            'date_range': {
                'start': date_range[0],
                'end': date_range[1]
            } if date_range[0] else None,
            'by_model_version': model_versions,
            'by_model_type': model_types
        }


def main():
    """Example usage of PredictionCollector."""
    import sys
    
    # Initialize collector
    collector = PredictionCollector()
    
    # Ingest logs from inference directory
    print("\nIngesting prediction logs...")
    ingested = collector.ingest_logs()
    print(f"✓ Ingested {ingested} predictions")
    
    # Get database statistics
    print("\nDatabase Statistics:")
    stats = collector.get_database_stats()
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Unique dates: {stats['unique_dates']}")
    print(f"  Unique items: {stats['unique_items']}")
    print(f"  Unique stores: {stats['unique_stores']}")
    print(f"  Date range: {stats['date_range']}")
    print(f"  By model version: {stats['by_model_version']}")
    print(f"  By model type: {stats['by_model_type']}")
    
    # Get recent predictions
    print("\nRecent Predictions (last 7 days):")
    recent = collector.get_recent_predictions(days=7)
    if len(recent) > 0:
        print(recent.head(10))
        
        # Show statistics
        print("\nStatistics by model version:")
        print(collector.get_prediction_stats(days=7))
    else:
        print("  No predictions found yet")


if __name__ == "__main__":
    main()
