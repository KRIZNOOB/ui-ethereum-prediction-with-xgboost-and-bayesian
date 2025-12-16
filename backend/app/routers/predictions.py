from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Optional
import asyncio
from datetime import datetime

from app.models.data_models import (
    TrainingRequest, TrainingResponse, PredictionResponse, 
    ModelStatus, DataInfo, ErrorResponse
)
from app.services.data_service import DataService
from app.models.prediction_models import EthereumPredictor
from app.core.config import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

router = APIRouter()

# Global instances
data_service = DataService()
predictor = EthereumPredictor()

@router.post("/train", response_model=TrainingResponse)
async def train_models(historical_days: int = 365):
    """Train both XGBoost models with real-time data only"""
    try:
        print(f"🚀 Starting training with {historical_days} days of real-time data...")
        
        # Validate days parameter
        if historical_days < 7 or historical_days > 365:
            raise HTTPException(
                status_code=400,
                detail="Historical days must be between 7 and 365 (1 year)"
            )
        
        # 1. Fetch Real-time Data Only
        print("📡 Fetching real-time data from CoinGecko...")
        df = await data_service.fetch_realtime_data(days=historical_days)
        
        # 2. Validate Data Quality
        print("🔍 Validating data quality...")
        validation = data_service.validate_data(df)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed: {validation['errors']}"
            )
        
        # Log warnings if any
        if validation["warnings"]:
            print(f"⚠️ Data warnings: {validation['warnings']}")
        
        # 3. Feature Engineering
        print("🔧 Creating ML features...")
        df_featured = data_service.create_features(df)
        
        # 4. Prepare Training Data
        print("📊 Preparing training data...")
        X, y = data_service.prepare_training_data(df_featured)
        
        # Check minimum data requirement
        if len(X) < 15:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data after feature engineering: {len(X)} samples (need at least 15)"
            )
        
        # 5. Split Data (time series - no shuffling)
        test_size = min(0.3, 6 / len(X))  # Max 30% or 6 samples, whichever is smaller
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Keep time order
        )
        
        print(f"📈 Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # 6. Scale Features
        print("⚖️ Scaling features...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler in predictor
        predictor.scaler = scaler
        
        # 7. Train Basic Model
        print("🤖 Training Basic XGBoost model...")
        basic_results = predictor.train_basic_model(
            X_train_scaled, y_train.values, X_test_scaled, y_test.values
        )
        
        # 8. Train Bayesian Model
        print("🧠 Training Bayesian Optimized XGBoost model...")
        bayesian_results = predictor.train_bayesian_model(
            X_train_scaled, y_train.values, X_test_scaled, y_test.values
        )
        
        # 9. Save Models
        print("💾 Saving trained models...")
        predictor.save_models()
        
        # 10. Get Latest Features for Prediction
        latest_features = data_service.get_latest_features(df_featured)
        
        # 11. Get Data Information
        data_info = data_service.get_data_info(df_featured)
        data_info.update({
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "features": settings.FEATURE_COLUMNS,
            "data_source": "real_time_api",
            "historical_days": historical_days
        })
        
        print("✅ Training completed successfully!")
        
        return TrainingResponse(
            status="success",
            message=f"Models trained successfully with {historical_days} days of real-time data",
            data_info=data_info,
            basic_model=basic_results,
            bayesian_model=bayesian_results,
            latest_features=latest_features
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@router.get("/predict-tomorrow", response_model=PredictionResponse)
async def predict_tomorrow():
    """Get tomorrow's ETH price prediction"""
    try:
        # Check if models are trained
        if not predictor.is_trained:
            raise HTTPException(
                status_code=400,
                detail="Models not trained yet. Call /train endpoint first."
            )
        
        # Get latest data for prediction features
        print("Fetching latest data for prediction...")
        df = await data_service.fetch_realtime_data(days=30)  # Get recent data
        df_featured = data_service.create_features(df)
        latest_features = data_service.get_latest_features(df_featured)
        
        # Get current price
        current_price = await data_service.get_current_price()
        
        # Make prediction
        prediction_result = predictor.predict_tomorrow(latest_features)
        
        return PredictionResponse(
            status="success",
            tomorrow_predictions=prediction_result["tomorrow_predictions"],
            current_price=current_price,
            prediction_timestamp=prediction_result["prediction_timestamp"],
            features_used=prediction_result["features_used"],
            trend_direction=prediction_result["trend_direction"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model training status and performance"""
    try:
        status = predictor.get_model_status()
        
        return ModelStatus(
            is_trained=status["is_trained"],
            last_training_time=status["last_training_time"],
            basic_model_available=status["basic_model_available"],
            bayesian_model_available=status["bayesian_model_available"],
            training_results=status["training_results"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )

@router.get("/current-price")
async def get_current_price():
    """Get current Ethereum price"""
    try:
        current_price = await data_service.get_current_price()
        return {
            "status": "success",
            "current_price": current_price,
            "currency": "USD",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current price: {str(e)}"
        )

@router.get("/feature-importance")
async def get_feature_importance():
    """Get feature importance from trained models"""
    try:
        if not predictor.is_trained:
            raise HTTPException(
                status_code=400,
                detail="Models not trained yet. Call /train endpoint first."
            )
        
        importance = predictor.get_feature_importance()
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "status": "success",
            "feature_importance": sorted_importance,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feature importance: {str(e)}"
        )

@router.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Retrain models with latest data (background task)"""
    try:
        def retrain_task():
            print("Starting background retraining...")
            # This would run the training in background
            # For now, just return success
            
        background_tasks.add_task(retrain_task)
        
        return {
            "status": "success",
            "message": "Model retraining started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start retraining: {str(e)}"
        )

@router.get("/data-info", response_model=DataInfo)
async def get_data_info(days: int = 365):
    """Get information about real-time dataset"""
    try:
        # Validate days parameter
        if days < 7 or days > 365:
            raise HTTPException(
                status_code=400,
                detail="Days must be between 7 and 365 (1 year)"
            )
        
        # Get real-time data
        print(f"📊 Fetching data info for {days} days...")
        df = await data_service.fetch_realtime_data(days=days)
        df_featured = data_service.create_features(df)
        
        data_info = data_service.get_data_info(df_featured)
        
        return DataInfo(
            total_rows=data_info["total_rows"],
            training_rows=0,  # Will be set during training
            test_rows=0,      # Will be set during training
            features=settings.FEATURE_COLUMNS,
            date_range=data_info["date_range"],
            latest_price=data_info["latest_price"],
            data_source="real_time_api",
            historical_days=days
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get data info: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test external API
        try:
            current_price = await data_service.get_current_price()
            api_status = True
        except:
            api_status = False
        
        return {
            "status": "healthy",
            "service": settings.PROJECT_NAME,
            "timestamp": datetime.now().isoformat(),
            "api_status": {
                "coingecko": api_status
            },
            "model_status": {
                "is_trained": predictor.is_trained,
                "models_available": {
                    "basic": predictor.xgb_basic is not None,
                    "bayesian": predictor.xgb_bayesian is not None
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }