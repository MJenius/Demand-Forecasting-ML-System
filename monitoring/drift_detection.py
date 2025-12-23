"""
Drift Detection Module - Step 7B

Detects when data or prediction distributions have shifted significantly,
indicating the model may need retraining.

Implements:
1. Prediction Drift - most important, no labels needed
2. Feature Drift - requires feature data

Start with Prediction Drift, extend to Feature Drift in Step 7C.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect prediction and feature drift using statistical tests."""
    
    def __init__(self, thresholds_path: str = "monitoring/thresholds.yaml"):
        """
        Initialize drift detector with thresholds.
        
        Args:
            thresholds_path: Path to thresholds YAML file
        """
        self.thresholds_path = Path(thresholds_path)
        self.thresholds = self._load_thresholds()
        self.reports_dir = Path("monitoring/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_thresholds(self) -> dict:
        """Load thresholds from YAML file."""
        try:
            with open(self.thresholds_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Thresholds file not found: {self.thresholds_path}")
            return self._default_thresholds()
    
    def _default_thresholds(self) -> dict:
        """Return default thresholds if file not found."""
        return {
            'prediction_drift': {
                'mean_shift_critical': 20,
                'mean_shift_warning': 10,
                'std_shift_critical': 30,
                'std_shift_warning': 15,
                'percentile_shift_warning': 15,
                'z_score_threshold': 2.5
            },
            'windows': {
                'reference_days': 14,
                'current_days': 7,
                'min_predictions': 50
            }
        }
    
    def detect_prediction_drift(self, collector, model_version: str = "v1") -> dict:
        """
        Detect prediction drift by comparing distributions.
        
        Compares:
        - Reference window: first N days (training era proxy)
        - Current window: last N days (recent behavior)
        
        Args:
            collector: PredictionCollector instance with prediction data
            model_version: Model version to analyze
        
        Returns:
            Drift report dictionary with status and metrics
        """
        config = self.thresholds['prediction_drift']
        windows = self.thresholds['windows']
        
        # Get all predictions for this model
        all_predictions = collector.get_predictions_by_model_version(model_version)
        
        if len(all_predictions) < windows['min_predictions']:
            return {
                'status': 'INSUFFICIENT_DATA',
                'timestamp': datetime.now().isoformat(),
                'model_version': model_version,
                'message': f'Only {len(all_predictions)} predictions found, need {windows["min_predictions"]}',
                'prediction_drift': {}
            }
        
        # Define windows
        max_date = pd.to_datetime(all_predictions['date']).max()
        current_start = max_date - timedelta(days=windows['current_days'])
        reference_end = max_date - timedelta(days=windows['current_days'])
        reference_start = reference_end - timedelta(days=windows['reference_days'])
        
        # Get reference and current windows
        reference_preds = all_predictions[
            (pd.to_datetime(all_predictions['date']) >= reference_start) &
            (pd.to_datetime(all_predictions['date']) <= reference_end)
        ]['prediction'].values
        
        current_preds = all_predictions[
            (pd.to_datetime(all_predictions['date']) >= current_start) &
            (pd.to_datetime(all_predictions['date']) <= max_date)
        ]['prediction'].values
        
        if len(reference_preds) < 10 or len(current_preds) < 10:
            return {
                'status': 'INSUFFICIENT_DATA',
                'timestamp': datetime.now().isoformat(),
                'model_version': model_version,
                'message': f'Reference: {len(reference_preds)}, Current: {len(current_preds)}',
                'prediction_drift': {}
            }
        
        # Calculate statistics
        ref_mean = np.mean(reference_preds)
        ref_std = np.std(reference_preds)
        curr_mean = np.mean(current_preds)
        curr_std = np.std(current_preds)
        
        # Calculate shifts
        mean_shift_pct = ((curr_mean - ref_mean) / (ref_mean + 1e-6)) * 100
        std_shift_pct = ((curr_std - ref_std) / (ref_std + 1e-6)) * 100
        
        # Calculate percentiles
        ref_p10, ref_p50, ref_p90 = np.percentile(reference_preds, [10, 50, 90])
        curr_p10, curr_p50, curr_p90 = np.percentile(current_preds, [10, 50, 90])
        
        p10_shift_pct = ((curr_p10 - ref_p10) / (ref_p10 + 1e-6)) * 100
        p50_shift_pct = ((curr_p50 - ref_p50) / (ref_p50 + 1e-6)) * 100
        p90_shift_pct = ((curr_p90 - ref_p90) / (ref_p90 + 1e-6)) * 100
        
        # Determine drift status
        status = self._classify_drift_status(
            mean_shift_pct, std_shift_pct, config
        )
        
        # Create report
        report = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'window': {
                'reference': {
                    'start': reference_start.strftime('%Y-%m-%d'),
                    'end': reference_end.strftime('%Y-%m-%d'),
                    'count': len(reference_preds)
                },
                'current': {
                    'start': current_start.strftime('%Y-%m-%d'),
                    'end': max_date.strftime('%Y-%m-%d'),
                    'count': len(current_preds)
                }
            },
            'prediction_drift': {
                'reference': {
                    'mean': float(ref_mean),
                    'std': float(ref_std),
                    'p10': float(ref_p10),
                    'p50': float(ref_p50),
                    'p90': float(ref_p90),
                    'min': float(np.min(reference_preds)),
                    'max': float(np.max(reference_preds))
                },
                'current': {
                    'mean': float(curr_mean),
                    'std': float(curr_std),
                    'p10': float(curr_p10),
                    'p50': float(curr_p50),
                    'p90': float(curr_p90),
                    'min': float(np.min(current_preds)),
                    'max': float(np.max(current_preds))
                },
                'shifts': {
                    'mean_pct': float(mean_shift_pct),
                    'std_pct': float(std_shift_pct),
                    'p10_pct': float(p10_shift_pct),
                    'p50_pct': float(p50_shift_pct),
                    'p90_pct': float(p90_shift_pct)
                },
                'thresholds': {
                    'mean_shift_critical': config['mean_shift_critical'],
                    'mean_shift_warning': config['mean_shift_warning'],
                    'std_shift_critical': config['std_shift_critical'],
                    'std_shift_warning': config['std_shift_warning']
                }
            }
        }
        
        return report
    
    def _classify_drift_status(self, mean_shift: float, std_shift: float, config: dict) -> str:
        """
        Classify drift status based on shifts.
        
        Args:
            mean_shift: Mean shift percentage
            std_shift: Std deviation shift percentage
            config: Configuration dictionary with thresholds
        
        Returns:
            Status: 'DRIFT', 'WARNING', or 'OK'
        """
        abs_mean_shift = abs(mean_shift)
        abs_std_shift = abs(std_shift)
        
        # Check critical thresholds
        if (abs_mean_shift > config['mean_shift_critical'] or 
            abs_std_shift > config['std_shift_critical']):
            return 'DRIFT'
        
        # Check warning thresholds
        if (abs_mean_shift > config['mean_shift_warning'] or 
            abs_std_shift > config['std_shift_warning']):
            return 'WARNING'
        
        return 'OK'
    
    def save_report(self, report: dict) -> Path:
        """
        Save drift report to JSON file.
        
        Args:
            report: Drift report dictionary
        
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.reports_dir / f"drift_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {report_path}")
        return report_path
    
    def print_report(self, report: dict) -> None:
        """Pretty-print drift report."""
        print("\n" + "="*70)
        print("PREDICTION DRIFT DETECTION REPORT")
        print("="*70)
        
        print(f"\nStatus: {report['status']}")
        print(f"Timestamp: {report.get('timestamp', 'N/A')}")
        print(f"Model Version: {report.get('model_version', 'N/A')}")
        
        if 'window' in report:
            print(f"\nAnalysis Window:")
            print(f"  Reference: {report['window']['reference']['start']} → {report['window']['reference']['end']}")
            print(f"            ({report['window']['reference']['count']} predictions)")
            print(f"  Current:   {report['window']['current']['start']} → {report['window']['current']['end']}")
            print(f"            ({report['window']['current']['count']} predictions)")
        
        if 'prediction_drift' in report and report['prediction_drift']:
            drift = report['prediction_drift']
            
            print(f"\nPrediction Distribution Shift:")
            print(f"\n  Reference Distribution:")
            print(f"    Mean: {drift['reference']['mean']:.4f}")
            print(f"    Std:  {drift['reference']['std']:.4f}")
            print(f"    Range: {drift['reference']['min']:.4f} - {drift['reference']['max']:.4f}")
            print(f"    Percentiles: p10={drift['reference']['p10']:.4f}, " +
                  f"p50={drift['reference']['p50']:.4f}, p90={drift['reference']['p90']:.4f}")
            
            print(f"\n  Current Distribution:")
            print(f"    Mean: {drift['current']['mean']:.4f}")
            print(f"    Std:  {drift['current']['std']:.4f}")
            print(f"    Range: {drift['current']['min']:.4f} - {drift['current']['max']:.4f}")
            print(f"    Percentiles: p10={drift['current']['p10']:.4f}, " +
                  f"p50={drift['current']['p50']:.4f}, p90={drift['current']['p90']:.4f}")
            
            print(f"\n  Shift Metrics:")
            shifts = drift['shifts']
            mean_shift_pct = shifts['mean_pct']
            std_shift_pct = shifts['std_pct']
            
            # Color-code the shifts based on severity
            mean_severity = "🔴 CRITICAL" if abs(mean_shift_pct) > drift['thresholds']['mean_shift_critical'] else \
                           "🟡 WARNING" if abs(mean_shift_pct) > drift['thresholds']['mean_shift_warning'] else \
                           "🟢 OK"
            std_severity = "🔴 CRITICAL" if abs(std_shift_pct) > drift['thresholds']['std_shift_critical'] else \
                          "🟡 WARNING" if abs(std_shift_pct) > drift['thresholds']['std_shift_warning'] else \
                          "🟢 OK"
            
            print(f"    Mean shift: {mean_shift_pct:+.2f}% {mean_severity}")
            print(f"    Std shift:  {std_shift_pct:+.2f}% {std_severity}")
            print(f"    P10 shift:  {shifts['p10_pct']:+.2f}%")
            print(f"    P50 shift:  {shifts['p50_pct']:+.2f}%")
            print(f"    P90 shift:  {shifts['p90_pct']:+.2f}%")
            
            print(f"\n  Decision Thresholds:")
            print(f"    Mean: DRIFT if |shift| > {drift['thresholds']['mean_shift_critical']}%, " +
                  f"WARNING if > {drift['thresholds']['mean_shift_warning']}%")
            print(f"    Std:  DRIFT if |shift| > {drift['thresholds']['std_shift_critical']}%, " +
                  f"WARNING if > {drift['thresholds']['std_shift_warning']}%")
        
        elif report['status'] == 'INSUFFICIENT_DATA':
            print(f"\nMessage: {report['message']}")
        
        print("\n" + "="*70)
        if report['status'] == 'DRIFT':
            print("⚠️  DRIFT DETECTED - Consider retraining the model")
        elif report['status'] == 'WARNING':
            print("⚠️  WARNING - Monitor closely for drift")
        else:
            print("✓ No drift detected - Model is performing normally")
        print("="*70 + "\n")


def main():
    """Example usage of DriftDetector."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from monitoring.collect_predictions import PredictionCollector
    
    # Initialize components
    print("Initializing drift detection...")
    detector = DriftDetector()
    collector = PredictionCollector()
    
    # Ingest logs if database is empty
    stats = collector.get_database_stats()
    if stats['total_predictions'] == 0:
        print("Database is empty. Ingesting prediction logs...")
        collector.ingest_logs()
        stats = collector.get_database_stats()
    
    # Get latest prediction statistics
    print("Gathering prediction statistics...")
    
    if stats['total_predictions'] == 0:
        print("No predictions found. Run inference or test collection first.")
        return
    
    print(f"Found {stats['total_predictions']} predictions")
    
    # Detect drift
    print("\nDetecting prediction drift...")
    report = detector.detect_prediction_drift(collector, model_version="v1")
    
    # Print report
    detector.print_report(report)
    
    # Save report
    report_path = detector.save_report(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
