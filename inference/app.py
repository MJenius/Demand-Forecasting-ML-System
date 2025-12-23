"""
FastAPI inference server for demand forecasting
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry.load_model import load_active_model
from inference.feature_builder import FeatureBuilder
from inference.router import PredictionRouter
from inference.logger import PredictionLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state - loaded at startup
model_bundle = None
feature_builder = None
prediction_router = None
prediction_logger = None


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    item_id: str
    store_id: str
    date: str  # YYYY-MM-DD format


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    item_id: str
    store_id: str
    date: str
    prediction: float
    model_used: str
    model_version: str


app = FastAPI(
    title="Demand Forecasting API",
    description="Production inference service for M5 Walmart demand forecasting",
    version="1.0"
)


@app.on_event("startup")
async def startup_event():
    """Load model and data at startup."""
    global model_bundle, feature_builder, prediction_router, prediction_logger
    
    logger.info("Loading model from registry...")
    model_bundle = load_active_model()
    
    logger.info(f"Model version: {model_bundle['version']}")
    logger.info(f"Features: {len(model_bundle['features'])} features")
    logger.info(f"SKU segments: {model_bundle['segments'].shape[0]} SKUs")
    
    # Load data for feature building
    logger.info("Loading data for feature engineering...")
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    # Load sales data
    sales_file = data_dir / "sales_long.parquet"
    if not sales_file.exists():
        raise FileNotFoundError(f"Sales data not found: {sales_file}")
    sales_data = pd.read_parquet(sales_file)
    logger.info(f"Loaded sales data: {len(sales_data)} rows")
    
    # Load calendar data
    calendar_file = Path(__file__).parent.parent / "data" / "raw" / "m5" / "calendar.csv"
    if not calendar_file.exists():
        raise FileNotFoundError(f"Calendar data not found: {calendar_file}")
    calendar_data = pd.read_csv(calendar_file)
    calendar_data['date'] = pd.to_datetime(calendar_data['date'])
    logger.info(f"Loaded calendar data: {len(calendar_data)} rows")
    
    # Load price data
    price_file = data_dir / "prices.parquet"
    if not price_file.exists():
        raise FileNotFoundError(f"Price data not found: {price_file}")
    price_data = pd.read_parquet(price_file)
    logger.info(f"Loaded price data: {len(price_data)} rows")
    
    # Initialize components
    feature_builder = FeatureBuilder(sales_data, calendar_data, price_data)
    prediction_router = PredictionRouter(
        model_bundle['model'],
        model_bundle['segments'],
        sales_data
    )
    prediction_logger = PredictionLogger()
    
    logger.info("✓ API startup complete")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_bundle is None:
        return {"status": "initializing"}
    
    return {
        "status": "healthy",
        "model_version": model_bundle['version'],
        "features": len(model_bundle['features'])
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for a given item, store, and date.
    
    Example request:
    {
        "item_id": "FOODS_3_090",
        "store_id": "CA_3",
        "date": "2016-04-25"
    }
    """
    
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. API still initializing.")
    
    try:
        # Build features for this prediction
        logger.info(f"Building features for {request.item_id}/{request.store_id}/{request.date}")
        features_dict = feature_builder.build_features(
            request.item_id,
            request.store_id,
            request.date,
            model_bundle['features']
        )
        
        # Route to appropriate model and get prediction
        prediction, model_used = prediction_router.predict(
            request.item_id,
            request.store_id,
            request.date,
            features_dict,
            model_bundle['features']
        )
        
        # Log the prediction
        prediction_logger.log_prediction(
            request.item_id,
            request.store_id,
            request.date,
            prediction,
            model_used,
            model_bundle['version'],
            features=features_dict
        )
        
        logger.info(f"Prediction: {prediction:.4f} (model: {model_used})")
        
        return PredictionResponse(
            item_id=request.item_id,
            store_id=request.store_id,
            date=request.date,
            prediction=prediction,
            model_used=model_used,
            model_version=model_bundle['version']
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/info")
async def model_info():
    """Get information about the loaded model."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "version": model_bundle['version'],
        "registered_at": model_bundle['metadata']['registered_at'],
        "features": model_bundle['features'],
        "metrics": model_bundle['metrics'],
        "segment_counts": model_bundle['segments']['is_normal_volume'].value_counts().to_dict()
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Demand Forecasting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
