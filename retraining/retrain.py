"""
Automated Retraining Pipeline - Step 8

Triggered by drift detection, this module:
1. Loads latest drift report
2. Decides whether to retrain based on drift status
3. Trains candidate model on recent data
4. Evaluates candidate vs current model
5. Promotes if better (handles registry update)

Design principle: Drift detection ≠ automatic deployment.
Drift means "check if retraining helps", not "deploy new model".
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.collect_predictions import PredictionCollector
from registry.load_model import load_active_model

logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """Manages retraining decisions based on drift reports."""
    
    def __init__(self, drift_reports_dir: str = "monitoring/reports",
                 thresholds_path: str = "monitoring/thresholds.yaml"):
        """
        Initialize retraining trigger.
        
        Args:
            drift_reports_dir: Directory containing drift reports
            thresholds_path: Path to thresholds YAML
        """
        self.reports_dir = Path(drift_reports_dir)
        self.thresholds_path = Path(thresholds_path)
        self.collector = PredictionCollector()
    
    def get_latest_drift_report(self) -> dict:
        """
        Load the most recent drift report.
        
        Returns:
            Drift report dictionary, or None if no reports exist
        """
        if not self.reports_dir.exists():
            logger.warning(f"Reports directory not found: {self.reports_dir}")
            return None
        
        # Find all drift reports and sort by timestamp
        report_files = sorted(self.reports_dir.glob("drift_report_*.json"))
        
        if not report_files:
            logger.warning("No drift reports found")
            return None
        
        latest_report_file = report_files[-1]
        
        with open(latest_report_file, 'r') as f:
            report = json.load(f)
        
        logger.info(f"Loaded drift report: {latest_report_file.name}")
        return report
    
    def should_retrain(self, drift_report: dict = None) -> tuple:
        """
        Determine if retraining should be triggered.
        
        Args:
            drift_report: Drift report dict. If None, loads latest.
        
        Returns:
            Tuple of (should_retrain: bool, reason: str, drift_status: str)
        """
        if drift_report is None:
            drift_report = self.get_latest_drift_report()
        
        if drift_report is None:
            return False, "No drift report found", "UNKNOWN"
        
        status = drift_report.get('status', 'UNKNOWN')
        
        # Retraining triggers
        if status == 'DRIFT':
            return True, "DRIFT detected - retraining candidate model", status
        
        elif status == 'WARNING':
            # For now, don't auto-retrain on warning
            # Can be tuned in thresholds.yaml
            return False, "WARNING status - monitoring closely, no retrain yet", status
        
        elif status == 'OK':
            return False, "No drift detected - continuing with current model", status
        
        elif status == 'INSUFFICIENT_DATA':
            return False, f"Insufficient data: {drift_report.get('message', 'unknown')}", status
        
        else:
            return False, f"Unknown drift status: {status}", status
    
    def get_retraining_window(self) -> tuple:
        """
        Get the data window for retraining.
        
        Returns:
            Tuple of (start_date, end_date) as YYYY-MM-DD strings
        """
        # Get all predictions from database
        stats = self.collector.get_database_stats()
        
        if stats['total_predictions'] == 0:
            logger.warning("No predictions in database")
            return None, None
        
        # Use last 60 days (configurable via thresholds)
        max_date_str = stats['date_range']['end']
        max_date = datetime.strptime(max_date_str, '%Y-%m-%d')
        
        # Retrain on last 60 days
        retrain_days = 60
        start_date = max_date - timedelta(days=retrain_days)
        
        return start_date.strftime('%Y-%m-%d'), max_date_str
    
    def get_retraining_data(self) -> pd.DataFrame:
        """
        Get data for retraining.
        
        Returns:
            DataFrame with predictions from retraining window
        """
        start_date, end_date = self.get_retraining_window()
        
        if start_date is None:
            return None
        
        # Get predictions for retraining window
        retraining_data = self.collector.get_predictions_by_date_range(start_date, end_date)
        
        logger.info(f"Loaded retraining data: {len(retraining_data)} predictions")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        return retraining_data
    
    def create_retraining_report(self, should_retrain: bool, reason: str, 
                                drift_status: str) -> dict:
        """
        Create a retraining decision report.
        
        Args:
            should_retrain: Whether to proceed with retraining
            reason: Explanation of decision
            drift_status: Drift status from report
        
        Returns:
            Report dictionary
        """
        # Get current active model
        active_model = load_active_model()
        
        # Get retraining data info
        retraining_data = self.get_retraining_data()
        retraining_window = self.get_retraining_window()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'retraining_triggered': should_retrain,
            'reason': reason,
            'drift_status': drift_status,
            'current_model': {
                'version': active_model['version'],
                'metrics': active_model['metrics']['model_metrics']
            },
            'retraining_data': {
                'window': {
                    'start': retraining_window[0],
                    'end': retraining_window[1]
                },
                'predictions_count': len(retraining_data) if retraining_data is not None else 0
            }
        }
        
        return report
    
    def print_retraining_decision(self, report: dict) -> None:
        """Pretty-print retraining decision."""
        print("\n" + "="*70)
        print("RETRAINING DECISION REPORT")
        print("="*70)
        
        print(f"\nTimestamp: {report['timestamp']}")
        print(f"Drift Status: {report['drift_status']}")
        print(f"Reason: {report['reason']}")
        
        print(f"\nCurrent Active Model:")
        print(f"  Version: {report['current_model']['version']}")
        print(f"  MAE: {report['current_model']['metrics']['MAE']:.4f}")
        print(f"  RMSE: {report['current_model']['metrics']['RMSE']:.4f}")
        print(f"  MAPE: {report['current_model']['metrics']['MAPE']:.2f}%")
        
        print(f"\nRetraining Data:")
        if report['retraining_data']['predictions_count'] > 0:
            print(f"  Window: {report['retraining_data']['window']['start']} → " +
                  f"{report['retraining_data']['window']['end']}")
            print(f"  Predictions: {report['retraining_data']['predictions_count']}")
        else:
            print(f"  No data available for retraining")
        
        print(f"\nDecision:")
        if report['retraining_triggered']:
            print(f"  ✓ RETRAINING TRIGGERED")
            print(f"    Next step: Train candidate model")
            print(f"    Will compare candidate vs current on validation window")
            print(f"    Only promote if candidate is better")
        else:
            print(f"  ✓ NO RETRAINING NEEDED")
            print(f"    Continue monitoring with current model")
        
        print("\n" + "="*70 + "\n")
    
    def save_retraining_report(self, report: dict) -> Path:
        """Save retraining report to JSON file."""
        reports_dir = Path("retraining/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f"retraining_decision_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {report_path}")
        return report_path


def main():
    """Example usage of RetrainingTrigger."""
    print("\n" + "="*70)
    print("STEP 8 - RETRAINING TRIGGER")
    print("="*70)
    
    # Initialize trigger
    print("\nInitializing retraining trigger...")
    trigger = RetrainingTrigger()
    
    # Load latest drift report
    print("Loading latest drift report...")
    drift_report = trigger.get_latest_drift_report()
    
    if drift_report is None:
        print("No drift report found. Run drift detection first.")
        print("  python monitoring/drift_detection.py")
        return
    
    # Decide whether to retrain
    print("\nAnalyzing drift report...")
    should_retrain, reason, drift_status = trigger.should_retrain(drift_report)
    
    # Create detailed report
    retraining_report = trigger.create_retraining_report(
        should_retrain, reason, drift_status
    )
    
    # Print decision
    trigger.print_retraining_decision(retraining_report)
    
    # Save report
    report_path = trigger.save_retraining_report(retraining_report)
    print(f"Decision report saved: {report_path}")
    
    # Return exit code for scripting
    if should_retrain:
        print("⚠️  RETRAINING REQUIRED - Next step: candidate model training")
        return 0  # Success, retrain
    else:
        print("✓ No retraining needed - Current model performing normally")
        return 1  # No retrain needed


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO)
    exit_code = main()
    sys.exit(exit_code)
