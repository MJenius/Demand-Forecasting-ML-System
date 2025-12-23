"""
Train candidate model for retraining pipeline.

This module trains a new LightGBM model using recent data to address
the drift detected in the current model.

Key design principles:
- Use SAME feature pipeline (no changes)
- Use SAME SKU segmentation (no changes)
- Use SAME hyperparameters (deterministic)
- Only the training data window changes
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from registry.load_model import load_active_model
from monitoring.collect_predictions import PredictionCollector

logger = logging.getLogger(__name__)


class CandidateModelTrainer:
    """Train candidate model on recent data after drift detection."""
    
    def __init__(self, retrain_days: int = 60):
        """
        Initialize candidate trainer.
        
        Args:
            retrain_days: Number of recent days to use for training
        """
        self.retrain_days = retrain_days
        self.collector = PredictionCollector()
        self.active_model = load_active_model()
        self.candidate_dir = Path("registry/models/candidate")
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
    
    def get_training_data(self, target_date: str = None) -> pd.DataFrame:
        """
        Get data for candidate training.
        
        Args:
            target_date: End date for training window (default: today)
        
        Returns:
            DataFrame with training data from features/features_v2.parquet
        """
        logger.info(f"Loading training data for candidate model...")
        
        # Load full featured dataset
        features_file = Path("features/features_v2.parquet")
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        df = pd.read_parquet(features_file)
        logger.info(f"Loaded {len(df)} rows from features_v2.parquet")
        
        # Determine date range for retraining
        if target_date is None:
            target_date = pd.to_datetime(df['date']).max()
        else:
            target_date = pd.to_datetime(target_date)
        
        start_date = target_date - timedelta(days=self.retrain_days)
        
        # Filter to retraining window
        df['date'] = pd.to_datetime(df['date'])
        df_retrain = df[
            (df['date'] >= start_date) & 
            (df['date'] <= target_date)
        ].copy()
        
        logger.info(f"Filtered to {len(df_retrain)} rows for retraining")
        logger.info(f"Training window: {start_date.date()} to {target_date.date()}")
        
        return df_retrain
    
    def prepare_training_data(self, df: pd.DataFrame, val_end_date: str = None) -> tuple:
        """
        Prepare training and validation data using same split as original.
        
        Uses time-based split: everything before val_end_date is train,
        everything after is validation.
        
        Args:
            df: Training data
            val_end_date: Date after which is validation (YYYY-MM-DD)
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if val_end_date is None:
            # Use last 7 days as validation
            max_date = pd.to_datetime(df['date']).max()
            val_end_date = max_date - timedelta(days=7)
        else:
            val_end_date = pd.to_datetime(val_end_date)
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Time-based split
        train = df[df['date'] <= val_end_date].copy()
        val = df[df['date'] > val_end_date].copy()
        
        # Remove rows with NaN features
        feature_names = self.active_model['features']
        train = train.dropna(subset=feature_names + ['target'])
        val = val.dropna(subset=feature_names + ['target'])
        
        X_train = train[feature_names].values
        y_train = train['target'].values
        X_val = val[feature_names].values
        y_val = val['target'].values
        
        logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        return X_train, y_train, X_val, y_val, feature_names, train, val
    
    def train_candidate_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             feature_names: list) -> dict:
        """
        Train LightGBM candidate model with SAME hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
        
        Returns:
            Dictionary with model and training info
        """
        logger.info("Training candidate LightGBM model...")
        
        # Use SAME hyperparameters as original model
        lightgbm_params = self.active_model['metadata']['lightgbm_params']
        
        # Create LGBMRegressor with same params
        model = lgb.LGBMRegressor(**lightgbm_params)
        
        # Train with validation set for monitoring
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae'
        )
        
        best_iteration = model.best_iteration_
        logger.info(f"Training complete. Best iteration: {best_iteration}")
        
        return {
            'model': model,
            'best_iteration': best_iteration,
            'hyperparams': lightgbm_params
        }
    
    def evaluate_model(self, model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """
        Evaluate model on validation set.
        
        Args:
            model: LightGBM model
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            Dictionary with metrics
        """
        # Make predictions
        preds = model.predict(X_val)
        preds = np.maximum(preds, 0)  # Clip to non-negative
        
        # Calculate metrics
        mae = np.mean(np.abs(y_val - preds))
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        mape = np.mean(np.abs((y_val - preds) / (y_val + 1e-6))) * 100
        
        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape)
        }
        
        logger.info(f"Validation metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        
        return metrics
    
    def save_candidate_model(self, model, metrics: dict, train_info: dict) -> Path:
        """
        Save candidate model to registry/models/candidate/.
        
        Args:
            model: LightGBM model
            metrics: Validation metrics
            train_info: Training information
        
        Returns:
            Path to candidate model directory
        """
        # Save model
        model_path = self.candidate_dir / "lgb_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved candidate model: {model_path}")
        
        # Save metadata
        metadata = {
            'model_version': 'candidate',
            'model_type': 'lightgbm',
            'trained_at': datetime.now().isoformat(),
            'training': {
                'data_window': train_info['data_window'],
                'train_samples': train_info['train_samples'],
                'val_samples': train_info['val_samples']
            },
            'lightgbm_params': train_info['hyperparams'],
            'features': self.active_model['metadata']['features']
        }
        
        with open(self.candidate_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save metrics
        metrics_data = {
            'candidate_metrics': metrics,
            'current_metrics': self.active_model['metrics']['model_metrics'],
            'comparison': {
                'MAE': {
                    'current': self.active_model['metrics']['model_metrics']['MAE'],
                    'candidate': metrics['MAE'],
                    'improvement_pct': (
                        (self.active_model['metrics']['model_metrics']['MAE'] - metrics['MAE']) /
                        self.active_model['metrics']['model_metrics']['MAE'] * 100
                    )
                },
                'RMSE': {
                    'current': self.active_model['metrics']['model_metrics']['RMSE'],
                    'candidate': metrics['RMSE'],
                    'improvement_pct': (
                        (self.active_model['metrics']['model_metrics']['RMSE'] - metrics['RMSE']) /
                        self.active_model['metrics']['model_metrics']['RMSE'] * 100
                    )
                }
            }
        }
        
        with open(self.candidate_dir / "metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved candidate metrics and metadata")
        
        return self.candidate_dir
    
    def print_candidate_summary(self, metrics: dict) -> None:
        """Pretty-print candidate model summary."""
        current = self.active_model['metrics']['model_metrics']
        
        print("\n" + "="*70)
        print("CANDIDATE MODEL TRAINING COMPLETE")
        print("="*70)
        
        print(f"\nCandidate Metrics:")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        
        print(f"\nCurrent Model (v{self.active_model['version']}):")
        print(f"  MAE:  {current['MAE']:.4f}")
        print(f"  RMSE: {current['RMSE']:.4f}")
        print(f"  MAPE: {current['MAPE']:.2f}%")
        
        print(f"\nImprovement:")
        mae_improvement = (current['MAE'] - metrics['MAE']) / current['MAE'] * 100
        rmse_improvement = (current['RMSE'] - metrics['RMSE']) / current['RMSE'] * 100
        
        print(f"  MAE:  {mae_improvement:+.2f}%")
        print(f"  RMSE: {rmse_improvement:+.2f}%")
        
        print(f"\nCandidate saved to: {self.candidate_dir}")
        print("Next step: Evaluate and decide on promotion\n")


def main():
    """Train candidate model."""
    print("\n" + "="*70)
    print("STEP 8B - CANDIDATE MODEL TRAINING")
    print("="*70 + "\n")
    
    try:
        # Initialize trainer
        trainer = CandidateModelTrainer(retrain_days=60)
        
        # Load training data
        print("Loading training data...")
        df_retrain = trainer.get_training_data()
        
        if len(df_retrain) == 0:
            print("ERROR: No training data available")
            return 1
        
        # Prepare train/val split
        print("\nPreparing training and validation sets...")
        X_train, y_train, X_val, y_val, feature_names, train_df, val_df = trainer.prepare_training_data(df_retrain)
        
        # Train candidate
        print("\nTraining candidate model...")
        candidate_info = trainer.train_candidate_model(X_train, y_train, X_val, y_val, feature_names)
        
        # Evaluate
        print("\nEvaluating candidate...")
        metrics = trainer.evaluate_model(candidate_info['model'], X_val, y_val)
        
        # Save
        print("\nSaving candidate model...")
        train_info = {
            'data_window': {
                'start': str(train_df['date'].min().date()),
                'end': str(train_df['date'].max().date())
            },
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'best_iteration': candidate_info['best_iteration'],
            'hyperparams': candidate_info['hyperparams']
        }
        
        trainer.save_candidate_model(candidate_info['model'], metrics, train_info)
        
        # Print summary
        trainer.print_candidate_summary(metrics)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during candidate training: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit_code = main()
    sys.exit(exit_code)
