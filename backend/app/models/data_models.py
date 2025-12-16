from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

# Request Models (Input to Frontend)
class TrainingRequest(BaseModel):
    """Request for training models"""
    use_realtime_data: bool = True
    historical_days: int = 365
    
class PredictionRequest(BaseModel):
    """Request for prediction"""
    features: Optional[Dict[str, float]] = None
    use_latest_data: bool = True

# Response Models (Output to Frontend)
class ModelMetrics(BaseModel):
    """Model performance metrics"""
    rmse: float
    mae: float
    r2: float
    mae_pct: float
    rmse_pct: float

class ModelTrainingResult(BaseModel):
    """Result from training a single model"""
    model_type: str
    training_time: float
    test_metrics: ModelMetrics
    train_metrics: ModelMetrics
    best_params: Optional[Dict[str, Any]] = None
    best_iteration: Optional[int] = None

class TrainingResponse(BaseModel):
    """Complete training response"""
    status: str
    message: Optional[str] = None
    data_info: Optional[Dict[str, Any]] = None
    basic_model: Optional[ModelTrainingResult] = None
    bayesian_model: Optional[ModelTrainingResult] = None
    latest_features: Optional[Dict[str, float]] = None

class PredictionResult(BaseModel):
    """Contains prediction results from models"""
    basic_model: Optional[float] = None
    bayesian_model: float
    recommended: float
    confidence_interval: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    status: str
    tomorrow_predictions: PredictionResult
    current_price: Optional[float] = None
    prediction_timestamp: str
    features_used: Dict[str, float]
    trend_direction: Optional[str] = None  # "bullish" or "bearish"

class ModelStatus(BaseModel):
    """Current model status"""
    is_trained: bool
    last_training_time: Optional[str] = None
    basic_model_available: bool = False
    bayesian_model_available: bool = False
    training_results: Optional[Dict[str, ModelTrainingResult]] = None

class DataInfo(BaseModel):
    """Dataset information"""
    total_rows: int
    training_rows: int
    test_rows: int
    features: List[str]
    date_range: Dict[str, str]
    latest_price: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: str
    api_status: Dict[str, bool]

# Error Models
class ErrorResponse(BaseModel):
    """Standard error response"""
    status: str = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None