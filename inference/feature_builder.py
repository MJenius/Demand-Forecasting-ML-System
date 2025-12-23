"""
Feature builder for inference - builds features for a single prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class FeatureBuilder:
    """Build features for a single SKU-date combination at inference time."""
    
    def __init__(self, sales_data: pd.DataFrame, calendar_data: pd.DataFrame, 
                 price_data: pd.DataFrame):
        """
        Initialize with historical data.
        
        Args:
            sales_data: Full sales_long format (item_id, store_id, date, sales)
            calendar_data: Calendar with dates and event columns
            price_data: Price data with (item_id, store_id, date, sell_price)
        """
        self.sales_data = sales_data.copy()
        self.calendar_data = calendar_data.copy()
        self.price_data = price_data.copy()
        
        # Ensure date columns are datetime
        self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
        self.calendar_data['date'] = pd.to_datetime(self.calendar_data['date'])
        self.price_data['date'] = pd.to_datetime(self.price_data['date'])
        
        # Create is_event flag if not present
        if 'is_event' not in self.calendar_data.columns:
            self.calendar_data['is_event'] = (
                self.calendar_data['event_name_1'].notna() | 
                self.calendar_data['event_name_2'].notna()
            ).astype(int)
        
    def build_features(self, item_id: str, store_id: str, target_date: str,
                      feature_names: list) -> dict:
        """
        Build features for a single prediction.
        
        Args:
            item_id: Item ID (e.g., "FOODS_3_090")
            store_id: Store ID (e.g., "CA_3")
            target_date: Date to predict for (YYYY-MM-DD)
            feature_names: List of features to compute
        
        Returns:
            Dictionary of feature_name -> value
        """
        target_date = pd.to_datetime(target_date)
        
        # Get historical data for this SKU up to target_date (exclusive)
        sku_mask = (self.sales_data['item_id'] == item_id) & \
                   (self.sales_data['store_id'] == store_id)
        sku_history = self.sales_data[sku_mask & (self.sales_data['date'] < target_date)].copy()
        sku_history = sku_history.sort_values('date').reset_index(drop=True)
        
        if len(sku_history) == 0:
            raise ValueError(f"No historical data for {item_id} in {store_id} before {target_date}")
        
        features = {}
        
        # Lag features
        for lag in [1, 7, 14, 21, 28]:
            lag_date = target_date - timedelta(days=lag)
            lag_val = sku_history[sku_history['date'] == lag_date]['sales'].values
            features[f'lag_{lag}'] = float(lag_val[0]) if len(lag_val) > 0 else np.nan
        
        # Rolling features
        for window in [7, 14, 28]:
            window_end = target_date - timedelta(days=1)
            window_start = window_end - timedelta(days=window-1)
            
            window_data = sku_history[(sku_history['date'] >= window_start) & 
                                     (sku_history['date'] <= window_end)]['sales']
            
            if len(window_data) > 0:
                features[f'rolling_mean_{window}'] = float(window_data.mean())
                features[f'rolling_std_{window}'] = float(window_data.std()) if len(window_data) > 1 else 0.0
                features[f'rolling_max_{window}'] = float(window_data.max())
                features[f'rolling_min_{window}'] = float(window_data.min())
            else:
                features[f'rolling_mean_{window}'] = np.nan
                features[f'rolling_std_{window}'] = np.nan
                features[f'rolling_max_{window}'] = np.nan
                features[f'rolling_min_{window}'] = np.nan
        
        # Price features
        price_mask = (self.price_data['item_id'] == item_id) & \
                     (self.price_data['store_id'] == store_id) & \
                     (self.price_data['date'] <= target_date)
        price_history = self.price_data[price_mask].sort_values('date')
        
        if len(price_history) > 0:
            current_price = float(price_history.iloc[-1]['sell_price'])
            features['sell_price'] = current_price
            
            # Price normalization (mean of all prices seen)
            mean_price = float(price_history['sell_price'].mean())
            features['price_norm'] = current_price / mean_price if mean_price > 0 else 1.0
            
            # Price change from 7 days ago
            price_7d_ago = price_history[price_history['date'] < target_date - timedelta(days=7)]
            if len(price_7d_ago) > 0:
                features['price_change'] = current_price - float(price_7d_ago.iloc[-1]['sell_price'])
            else:
                features['price_change'] = 0.0
            
            features['is_price_reduced'] = 1.0 if features['price_change'] < 0 else 0.0
        else:
            features['sell_price'] = np.nan
            features['price_norm'] = np.nan
            features['price_change'] = np.nan
            features['is_price_reduced'] = np.nan
        
        # Calendar features
        calendar_row = self.calendar_data[self.calendar_data['date'] == target_date]
        
        features['day_of_week'] = float(target_date.dayofweek)
        features['week_of_year'] = float(target_date.isocalendar().week)
        features['is_weekend'] = 1.0 if target_date.dayofweek >= 5 else 0.0
        features['month'] = float(target_date.month)
        features['day_of_month'] = float(target_date.day)
        features['is_month_end'] = 1.0 if target_date.day >= 26 else 0.0
        features['is_month_start'] = 1.0 if target_date.day <= 5 else 0.0
        
        if len(calendar_row) > 0:
            features['is_event'] = float(calendar_row.iloc[0]['is_event'])
        else:
            features['is_event'] = 0.0
        
        # Filter to only requested features
        return {k: v for k, v in features.items() if k in feature_names}
