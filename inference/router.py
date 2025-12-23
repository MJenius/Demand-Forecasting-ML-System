"""
Prediction routing logic - routes to LightGBM or 7-Day MA based on SKU segment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class PredictionRouter:
    """Route predictions to the correct model based on SKU segment."""
    
    def __init__(self, lgb_model, sku_segments: pd.DataFrame, 
                 sales_data: pd.DataFrame):
        """
        Initialize router with model and segmentation data.
        
        Args:
            lgb_model: Trained LightGBM model
            sku_segments: DataFrame with columns [item_id, store_id, segment]
            sales_data: Full sales data for 7-Day MA fallback
        """
        self.lgb_model = lgb_model
        self.sku_segments = sku_segments.copy()
        self.sales_data = sales_data.copy()
        
        # Create index for fast segment lookup
        self.segment_dict = {}
        for _, row in self.sku_segments.iterrows():
            key = (row['item_id'], row['store_id'])
            # Convert boolean to segment name
            self.segment_dict[key] = 'normal_volume' if row['is_normal_volume'] else 'low_volume'
    
    def get_segment(self, item_id: str, store_id: str) -> str:
        """
        Get the segment for a given SKU.
        
        Args:
            item_id: Item ID
            store_id: Store ID
        
        Returns:
            Segment name: 'normal_volume' or 'low_volume'
        """
        key = (item_id, store_id)
        if key in self.segment_dict:
            return self.segment_dict[key]
        # Default to low_volume if not in registry
        return 'low_volume'
    
    def predict(self, item_id: str, store_id: str, target_date: str,
               features_dict: dict, feature_names: list) -> tuple:
        """
        Route to appropriate model and return prediction.
        
        Args:
            item_id: Item ID
            store_id: Store ID
            target_date: Target date (YYYY-MM-DD)
            features_dict: Dictionary of computed features
            feature_names: Ordered list of feature names for model
        
        Returns:
            Tuple of (prediction, model_used)
        """
        segment = self.get_segment(item_id, store_id)
        
        if segment == 'normal_volume':
            return self._predict_lgb(features_dict, feature_names), 'lightgbm'
        else:
            return self._predict_ma7(item_id, store_id, target_date), 'moving_average'
    
    def _predict_lgb(self, features_dict: dict, feature_names: list) -> float:
        """
        Use LightGBM to make prediction.
        
        Args:
            features_dict: Dictionary of computed features
            feature_names: Ordered list of feature names
        
        Returns:
            Prediction value
        """
        # Create feature array in correct order
        feature_array = np.array([features_dict.get(f, np.nan) for f in feature_names]).reshape(1, -1)
        
        # Make prediction
        pred = self.lgb_model.predict(feature_array)[0]
        
        # Ensure non-negative
        return max(0.0, float(pred))
    
    def _predict_ma7(self, item_id: str, store_id: str, target_date: str) -> float:
        """
        Use 7-Day Moving Average for prediction.
        
        Args:
            item_id: Item ID
            store_id: Store ID
            target_date: Target date (YYYY-MM-DD)
        
        Returns:
            7-day moving average of past sales
        """
        target_date = pd.to_datetime(target_date)
        
        # Get historical data up to 7 days before target date
        sku_mask = (self.sales_data['item_id'] == item_id) & \
                   (self.sales_data['store_id'] == store_id)
        sku_history = self.sales_data[sku_mask].copy()
        sku_history['date'] = pd.to_datetime(sku_history['date'])
        
        window_end = target_date - timedelta(days=1)
        window_start = window_end - timedelta(days=6)
        
        window_data = sku_history[(sku_history['date'] >= window_start) & 
                                  (sku_history['date'] <= window_end)]['sales']
        
        if len(window_data) > 0:
            return float(window_data.mean())
        else:
            return 0.0
