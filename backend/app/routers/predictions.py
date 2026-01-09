from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Optional
import asyncio
from datetime import datetime, timedelta

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
    """Train both XGBoost models with real-time data - EXACT MATCH WITH PUBLISHED ARTICLE"""
    try:
        # Validate days parameter
        if historical_days < 90:
            raise HTTPException(
                status_code=400,
                detail="Historical days must be at least 90 for training"
            )
        
        # 1. Fetch Real-time Data
        df = await data_service.fetch_realtime_data(days=historical_days)
        
        # 2. Validate Data Quality
        validation = data_service.validate_data(df)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed: {validation['errors']}"
            )
        
        # 3. Feature Engineering
        df_featured = data_service.create_features(df)
        
        # 4. Prepare Training Data
        X, y = data_service.prepare_training_data(df_featured)
        
        # Check minimum data requirement
        if len(X) < 15:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data after feature engineering: {len(X)} samples (need at least 15)"
            )
        
        # 5. SPLIT DATA (80-20, NO SHUFFLE - TIME SERIES)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # 6. STANDARDIZE FEATURES
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler in predictor for future predictions
        predictor.scaler = scaler
        
        # 7. TRAIN BASIC XGBOOST
        import time
        start_time = time.time()
        
        basic_results = predictor.train_basic_model(
            X_train_scaled, y_train.values, 
            X_test_scaled, y_test.values
        )
        
        basic_training_time = time.time() - start_time
        basic_results["training_time"] = basic_training_time
        
        # 8. TRAIN BAYESIAN XGBOOST
        start_time = time.time()
        
        bayesian_results = predictor.train_bayesian_model(
            X_train_scaled, y_train.values,
            X_test_scaled, y_test.values
        )
        
        bayesian_training_time = time.time() - start_time
        bayesian_results["training_time"] = bayesian_training_time
        
        # 9. Save Models
        predictor.save_models()
        
        # 10. Get Latest Features
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
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@router.get("/predict-tomorrow", response_model=PredictionResponse)
async def predict_tomorrow():
    """Get tomorrow's price prediction"""
    try:
        # Check if model is trained
        if not predictor.is_trained:
            raise HTTPException(
                status_code=400,
                detail="Model not trained yet. Please train the model first."
            )
        
        # Get latest features
        df = await data_service.fetch_realtime_data(days=30)
        df_featured = data_service.create_features(df)
        latest_features = data_service.get_latest_features(df_featured)
        
        # Get prediction
        result = predictor.predict_tomorrow(latest_features)
        
        return PredictionResponse(
            status="success",
            tomorrow_predictions=result["tomorrow_predictions"],
            current_price=float(result["current_price"]),
            trend_direction=result["trend_direction"],
            price_change_pct=float(result["price_change_pct"]),
            prediction_timestamp=result["prediction_timestamp"],
            features_used=result["features_used"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model training status and performance"""
    try:
        status = predictor.get_model_status()
        
        return {
            "status": "success",
            "is_trained": status["is_trained"],
            "last_training_time": status["last_training_time"],
            "basic_model_available": status["basic_model_available"],
            "bayesian_model_available": status["bayesian_model_available"],
            "training_results": status["training_results"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )

@router.get("/current-price")
async def get_current_price():
    """Get current Ethereum price"""
    try:
        result = await data_service.get_current_price()
        
        return {
            "current_price": result["current_price"],
            "price_change_24h": result["price_change_24h"],
            "volume_24h": result["volume_24h"],
            "timestamp": result["timestamp"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current price: {str(e)}"
        )

@router.get("/historical-prices")
async def get_historical_prices(days: int = 30, interval: str = "daily"):
    try:
        if interval == "hourly":
            if days != 1:
                raise HTTPException(status_code=400, detail="Hourly data is only available for 1 day")
            
            import requests
            url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
            
            params = {
                "vs_currency": "usd",
                "days": "2"
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])
                
                historical_data = []
                for price_point in prices:
                    try:
                        timestamp = price_point[0]
                        price = price_point[1]
                        
                        dt = datetime.fromtimestamp(timestamp / 1000)
                        
                        historical_data.append({
                            "date": dt.isoformat(),
                            "price": round(float(price), 2),
                            "timestamp": timestamp
                        })
                    except (IndexError, ValueError, TypeError):
                        continue
                
                historical_data.sort(key=lambda x: x["timestamp"])
                last_24_hours = historical_data[-24:] if len(historical_data) >= 24 else historical_data
                
                return {
                    "status": "success",
                    "data": last_24_hours,
                    "interval": "hourly", 
                    "days": 1,
                    "total_points": len(last_24_hours),
                    "source": "coingecko"
                }
            else:
                raise HTTPException(status_code=500, detail=f"CoinGecko API error: {response.status_code}")
        
        else:  # daily data
            df = await data_service.fetch_realtime_data(days=days)
            
            historical_data = []
            for _, row in df.iterrows():
                try:
                    date_value = row['Date']    
                    price_value = row['Price']  
                    
                    if hasattr(date_value, 'strftime'):
                        date_str = date_value.strftime("%Y-%m-%d")
                    else:
                        date_str = str(date_value)[:10]
                    
                    historical_data.append({
                        "date": date_str,
                        "price": round(float(price_value), 2),
                        "timestamp": int(date_value.timestamp() * 1000) if hasattr(date_value, 'timestamp') else 0
                    })
                except Exception:
                    continue
            
            historical_data.sort(key=lambda x: x.get("timestamp", 0))
            
            return {
                "status": "success",
                "data": historical_data,
                "interval": "daily",
                "days": days,
                "total_points": len(historical_data),
                "source": "data_service"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical prices: {str(e)}")
    
@router.get("/feature-importance")
async def get_feature_importance():
    """Get feature importance from trained models"""
    try:
        if not predictor.is_trained or predictor.xgb_bayesian is None:
            raise HTTPException(
                status_code=400,
                detail="Model not trained yet"
            )
        
        # Get feature importance from Bayesian model
        importance = predictor.xgb_bayesian.feature_importances_
        feature_names = settings.FEATURE_COLUMNS
        
        # Sort by importance
        importance_dict = [
            {"feature": name, "importance": float(imp * 100)}
            for name, imp in zip(feature_names, importance)
        ]
        importance_dict.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "status": "success",
            "features": importance_dict,
            "model": "bayesian_xgboost"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feature importance: {str(e)}"
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
        df = await data_service.fetch_realtime_data(days=days)
        df_featured = data_service.create_features(df)
        
        data_info = data_service.get_data_info(df_featured)
        
        return DataInfo(
            total_rows=data_info["total_rows"],
            training_rows=0,
            test_rows=0,
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