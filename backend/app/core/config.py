from typing import List

class Settings:
    # Project Info
    PROJECT_NAME: str = "Ethereum Prediction System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # External APIs
    COINGECKO_API_URL: str = "https://api.coingecko.com/api/v3"
    
    # ML Model Settings
    MODEL_RETRAIN_INTERVAL: int = 86400  # 24 hours = 1 call per day
    DATA_FETCH_INTERVAL: int = 300       # 5 minutes (untuk current price)
    MAX_DAILY_TRAINING_CALLS: int = 5
    
    # Data Configuration
    HISTORICAL_DAYS: int = 365  
    TEST_SIZE: float = 0.2              # 20% for testing
    
    # XGBoost Basic Parameters
    XGB_BASIC_PARAMS = {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'max_depth': 2,
        'learning_rate': 0.3,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
    }
    
    # Bayesian Optimization Parameters
    BAYESIAN_SEARCH_PARAMS = {
        'n_estimators': (300, 500),
        'max_depth': (3, 5),
        'learning_rate': (0.05, 0.1),
        'subsample': (0.7, 0.9),
        'colsample_bytree': (0.7, 0.9),
    }
    
    # Feature Names
    FEATURE_COLUMNS: List[str] = [
        "Price_lag1", "Price_lag2", "Vol_lag1", 
        "Open_lag1", "High_lag1", "Low_lag1", 
        "MA3", "MA5"
    ]
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # File Paths
    MODEL_SAVE_PATH: str = "saved_models"
    LOG_FILE_PATH: str = "logs"

# Create global settings instance
settings = Settings()