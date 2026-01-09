import pandas as pd
import numpy as np
import time
import joblib
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from app.core.config import settings

class EthereumPredictor:
    def __init__(self):
        self.xgb_basic = None
        self.xgb_bayesian = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_results = {}
        self.last_training_time = None
    
    def train_basic_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        basic_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 2,
            'learning_rate': 0.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
        }
        
        self.xgb_basic = XGBRegressor(**basic_params)
        
        start_time = time.time()
        self.xgb_basic.fit(X_train, y_train)
        training_time_basic = time.time() - start_time
        
        y_pred_basic = self.xgb_basic.predict(X_test)
        y_train_pred_basic = self.xgb_basic.predict(X_train)
        
        mse_basic = mean_squared_error(y_test, y_pred_basic)
        mae_basic = mean_absolute_error(y_test, y_pred_basic)
        r2_basic = r2_score(y_test, y_pred_basic)
        rmse_basic = np.sqrt(mse_basic)
        
        mse_train_basic = mean_squared_error(y_train, y_train_pred_basic)
        mae_train_basic = mean_absolute_error(y_train, y_train_pred_basic)
        r2_train_basic = r2_score(y_train, y_train_pred_basic)
        rmse_train_basic = np.sqrt(mse_train_basic)
        
        y_range = y_test.max() - y_test.min()
        mae_pct_basic = (mae_basic / y_range) * 100 if y_range > 0 else 0
        rmse_pct_basic = (rmse_basic / y_range) * 100 if y_range > 0 else 0
        
        y_range_train = y_train.max() - y_train.min()
        mae_pct_train = (mae_train_basic / y_range_train) * 100 if y_range_train > 0 else 0
        rmse_pct_train = (rmse_train_basic / y_range_train) * 100 if y_range_train > 0 else 0
        
        results = {
            "model_type": "basic",
            "training_time": training_time_basic,
            "test_metrics": {
                "rmse": float(rmse_basic),
                "mae": float(mae_basic),
                "r2": float(r2_basic),
                "mae_pct": float(mae_pct_basic),
                "rmse_pct": float(rmse_pct_basic)
            },
            "train_metrics": {
                "rmse": float(rmse_train_basic),
                "mae": float(mae_train_basic),
                "r2": float(r2_train_basic),
                "mae_pct": float(mae_pct_train),
                "rmse_pct": float(rmse_pct_train)
            },
            "best_params": None,
            "best_iteration": None
        }
        
        self.training_results['basic'] = results
        return results

    def train_bayesian_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        search_spaces = {
            'n_estimators': Integer(300, 500),
            'max_depth': Integer(3, 5),
            'learning_rate': Real(0.05, 0.1, prior='log-uniform'),
            'subsample': Real(0.7, 0.9),
            'colsample_bytree': Real(0.7, 0.9),
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        opt = BayesSearchCV(
            estimator=XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                verbosity=0,
                n_jobs=-1
            ),
            search_spaces=search_spaces,
            cv=tscv,
            n_iter=25,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        start_time = time.time()
        opt.fit(X_train, y_train)
        training_time_bayesian = time.time() - start_time
        
        self.xgb_bayesian = opt.best_estimator_
        
        y_pred_bayesian = self.xgb_bayesian.predict(X_test)
        y_train_pred_bayes = self.xgb_bayesian.predict(X_train)
        
        mse_bayes = mean_squared_error(y_test, y_pred_bayesian)
        mae_bayes = mean_absolute_error(y_test, y_pred_bayesian)
        r2_bayes = r2_score(y_test, y_pred_bayesian)
        rmse_bayes = np.sqrt(mse_bayes)
        
        mse_train_bayes = mean_squared_error(y_train, y_train_pred_bayes)
        mae_train_bayes = mean_absolute_error(y_train, y_train_pred_bayes)
        r2_train_bayes = r2_score(y_train, y_train_pred_bayes)
        rmse_train_bayes = np.sqrt(mse_train_bayes)
        
        y_range = y_test.max() - y_test.min()
        mae_pct = (mae_bayes / y_range) * 100 if y_range > 0 else 0
        rmse_pct = (rmse_bayes / y_range) * 100 if y_range > 0 else 0
        
        y_range_train = y_train.max() - y_train.min()
        mae_pct_train = (mae_train_bayes / y_range_train) * 100 if y_range_train > 0 else 0
        rmse_pct_train = (rmse_train_bayes / y_range_train) * 100 if y_range_train > 0 else 0
        
        results_cv = opt.cv_results_
        best_index = np.argmax(results_cv['mean_test_score'])
        best_iteration = best_index + 1
        
        results = {
            "model_type": "bayesian",
            "training_time": training_time_bayesian,
            "best_params": opt.best_params_,
            "best_iteration": best_iteration,
            "test_metrics": {
                "rmse": float(rmse_bayes),
                "mae": float(mae_bayes),
                "r2": float(r2_bayes),
                "mae_pct": float(mae_pct),
                "rmse_pct": float(rmse_pct)
            },
            "train_metrics": {
                "rmse": float(rmse_train_bayes),
                "mae": float(mae_train_bayes),
                "r2": float(r2_train_bayes),
                "mae_pct": float(mae_pct_train),
                "rmse_pct": float(rmse_pct_train)
            }
        }
        
        self.training_results['bayesian'] = results
        self.is_trained = True
        self.last_training_time = datetime.now().isoformat()
        return results
    
    def predict_tomorrow(self, latest_features: Dict[str, float]) -> Dict:

        if not self.is_trained or self.xgb_bayesian is None:
            raise Exception("Models not trained yet. Call train_models first.")
        
        feature_array = np.array([
            [
            latest_features['Price_lag1'],
            latest_features['Price_lag2'], 
            latest_features['Vol_lag1'],
            latest_features['Open_lag1'],
            latest_features['High_lag1'],
            latest_features['Low_lag1'],
            latest_features['MA3'],
            latest_features['MA5']
        ]])
        
        feature_scaled = self.scaler.transform(feature_array)
        
        basic_pred = None
        if self.xgb_basic is not None:
            basic_pred = float(self.xgb_basic.predict(feature_scaled)[0])
            
        bayesian_pred = float(self.xgb_bayesian.predict(feature_scaled)[0])
        
        current_price = latest_features.get('Price_lag1', 0)
        trend_direction = "bullish" if bayesian_pred > current_price else "bearish"
        price_change_pct = ((bayesian_pred - current_price) / current_price * 100) if current_price > 0 else 0
        
        return {
            "tomorrow_predictions": {
                "basic_model": basic_pred,
                "bayesian_model": bayesian_pred,
                "recommended": bayesian_pred,
                "confidence_interval": {
                    "lower": bayesian_pred * 0.95,
                    "upper": bayesian_pred * 1.05
                }
            },
            "current_price": current_price,
            "prediction_timestamp": datetime.now().isoformat(),
            "features_used": latest_features,
            "trend_direction": trend_direction,
            "price_change_pct": round(price_change_pct, 2)
        }
    
    def save_models(self, filepath: str = None):
        """Save trained models"""
        if filepath is None:
            filepath = settings.MODEL_SAVE_PATH
        os.makedirs(filepath, exist_ok=True)
        
        if self.xgb_basic:
            joblib.dump(self.xgb_basic, f"{filepath}/xgb_basic.pkl")
        if self.xgb_bayesian:
            joblib.dump(self.xgb_bayesian, f"{filepath}/xgb_bayesian.pkl")
        joblib.dump(self.scaler, f"{filepath}/scaler.pkl")
        joblib.dump(self.training_results, f"{filepath}/training_results.pkl")
    
    def load_models(self, filepath: str = None):
        """Load trained models"""
        if filepath is None:
            filepath = settings.MODEL_SAVE_PATH
        
        try:
            if os.path.exists(f"{filepath}/xgb_basic.pkl"):
                self.xgb_basic = joblib.load(f"{filepath}/xgb_basic.pkl")
            if os.path.exists(f"{filepath}/xgb_bayesian.pkl"):
                self.xgb_bayesian = joblib.load(f"{filepath}/xgb_bayesian.pkl")
            if os.path.exists(f"{filepath}/scaler.pkl"):
                self.scaler = joblib.load(f"{filepath}/scaler.pkl")
            if os.path.exists(f"{filepath}/training_results.pkl"):
                self.training_results = joblib.load(f"{filepath}/training_results.pkl")
            
            self.is_trained = True
        except Exception as e:
            self.is_trained = False
    
    def get_model_status(self) -> Dict:
        """Get current model status"""
        return {
            "is_trained": self.is_trained,
            "last_training_time": self.last_training_time,
            "basic_model_available": self.xgb_basic is not None,
            "bayesian_model_available": self.xgb_bayesian is not None,
            "training_results": self.training_results
        }