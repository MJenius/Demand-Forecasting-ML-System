"""
Evaluate candidate model and decide on promotion.

This module compares candidate model performance against current active model.

Decision logic:
- PROMOTE if: candidate.MAE < current.MAE AND candidate.RMSE <= current.RMSE + 0.05
- REJECT otherwise with reason
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from registry.load_model import load_active_model
import joblib

logger = logging.getLogger(__name__)


class CandidateEvaluator:
    """Evaluate candidate model against current active model."""
    
    PROMOTION_THRESHOLD = 0.05  # RMSE tolerance
    
    def __init__(self):
        """Initialize evaluator."""
        self.current_model = load_active_model()
        self.candidate_dir = Path("registry/models/candidate")
        self.reports_dir = Path("retraining/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_candidate_metrics(self) -> dict:
        """
        Load candidate model metrics from saved file.
        
        Returns:
            Dictionary with candidate and comparison metrics
        """
        metrics_file = self.candidate_dir / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Candidate metrics not found: {metrics_file}")
        
        with open(metrics_file) as f:
            metrics_data = json.load(f)
        
        return metrics_data
    
    def evaluate(self) -> dict:
        """
        Evaluate candidate model.
        
        Returns:
            Dictionary with evaluation decision and details
        """
        logger.info("Evaluating candidate model...")
        
        # Load metrics
        metrics_data = self.load_candidate_metrics()
        candidate = metrics_data['candidate_metrics']
        current = metrics_data['current_metrics']
        comparison = metrics_data['comparison']
        
        # Decision logic
        decision = "REJECT"
        reason = ""
        
        mae_improvement = comparison['MAE']['improvement_pct']
        rmse_improvement = comparison['RMSE']['improvement_pct']
        rmse_current = current['RMSE']
        rmse_candidate = candidate['RMSE']
        rmse_diff = rmse_current - rmse_candidate
        
        logger.info(f"MAE improvement: {mae_improvement:+.2f}%")
        logger.info(f"RMSE improvement: {rmse_improvement:+.2f}%")
        logger.info(f"RMSE difference: {rmse_diff:+.4f}")
        
        # Check promotion criteria
        if candidate['MAE'] < current['MAE']:
            if rmse_candidate <= rmse_current + self.PROMOTION_THRESHOLD:
                decision = "PROMOTE"
                reason = f"MAE improved {mae_improvement:+.2f}%, RMSE within tolerance ({rmse_diff:+.4f})"
            else:
                decision = "REJECT"
                reason = f"RMSE degradation too large ({rmse_diff:+.4f} > {self.PROMOTION_THRESHOLD})"
        else:
            decision = "REJECT"
            reason = f"MAE not improved ({mae_improvement:.2f}%)"
        
        # Build evaluation report
        evaluation = {
            'decision': decision,
            'reason': reason,
            'evaluation_date': datetime.now().isoformat(),
            'current_model': f"v{self.current_model['version']}",
            'current_metrics': current,
            'candidate_metrics': candidate,
            'comparison': comparison,
            'promotion_threshold_rmse': self.PROMOTION_THRESHOLD
        }
        
        return evaluation
    
    def save_evaluation_report(self, evaluation: dict) -> Path:
        """
        Save evaluation report to JSON.
        
        Args:
            evaluation: Evaluation results
        
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"evaluation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info(f"Saved evaluation report: {report_file}")
        
        return report_file
    
    def print_evaluation_decision(self, evaluation: dict) -> None:
        """Pretty-print evaluation decision."""
        decision = evaluation['decision']
        current = evaluation['current_metrics']
        candidate = evaluation['candidate_metrics']
        comparison = evaluation['comparison']
        
        print("\n" + "="*70)
        print("CANDIDATE MODEL EVALUATION")
        print("="*70)
        
        print(f"\nCurrent Model (v{evaluation['current_model']}):")
        print(f"  MAE:  {current['MAE']:.4f}")
        print(f"  RMSE: {current['RMSE']:.4f}")
        print(f"  MAPE: {current['MAPE']:.2f}%")
        
        print(f"\nCandidate Model:")
        print(f"  MAE:  {candidate['MAE']:.4f}")
        print(f"  RMSE: {candidate['RMSE']:.4f}")
        print(f"  MAPE: {candidate['MAPE']:.2f}%")
        
        print(f"\nComparison:")
        print(f"  MAE improvement:  {comparison['MAE']['improvement_pct']:+.2f}%")
        print(f"  RMSE improvement: {comparison['RMSE']['improvement_pct']:+.2f}%")
        
        print(f"\nDecision: {decision}")
        print(f"Reason: {evaluation['reason']}")
        
        if decision == "PROMOTE":
            print(f"\n✓ Candidate is ready for promotion to v{int(evaluation['current_model'].split('v')[1]) + 1}")
            print("Next step: Promote candidate to active registry\n")
        else:
            print(f"\n✗ Candidate rejected. Current v{evaluation['current_model']} remains active")
            print("Next step: Investigate why retraining didn't help\n")
        
        print("="*70 + "\n")


def main():
    """Evaluate candidate model."""
    print("\n" + "="*70)
    print("STEP 8D - CANDIDATE MODEL EVALUATION")
    print("="*70 + "\n")
    
    try:
        # Initialize evaluator
        evaluator = CandidateEvaluator()
        
        # Check if candidate exists
        if not (evaluator.candidate_dir / "metrics.json").exists():
            print("ERROR: Candidate model not found. Run train_candidate.py first.")
            return 1
        
        # Evaluate candidate
        print("Evaluating candidate model...")
        evaluation = evaluator.evaluate()
        
        # Save report
        print("Saving evaluation report...")
        evaluator.save_evaluation_report(evaluation)
        
        # Print decision
        evaluator.print_evaluation_decision(evaluation)
        
        # Exit code indicates decision
        return 0 if evaluation['decision'] == "PROMOTE" else 1
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit_code = main()
    sys.exit(exit_code)
