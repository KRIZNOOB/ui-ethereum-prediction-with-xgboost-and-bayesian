import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.core.config import settings

class DataService:
    def __init__(self):
        self.features = settings.FEATURE_COLUMNS
        self.df = None
    
    async def fetch_realtime_data(self, days: int = None) -> pd.DataFrame:
        """Fetch real-time Ethereum data from CoinGecko API"""
        if days is None:
            days = settings.HISTORICAL_DAYS
            
        # Validate days parameter
        if days < 7 or days > 365:
            raise Exception("Days must be between 7 and 365")
            
        url = f"{settings.COINGECKO_API_URL}/coins/ethereum/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract prices and volumes
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                raise Exception("No price data received from API")
            
            # Convert to DataFrame format
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                date = datetime.fromtimestamp(timestamp / 1000)
                volume = volumes[i][1] if i < len(volumes) else 0
                
                # For daily data, use price as OHLC
                # In production, you'd get actual OHLC data
                df_data.append({
                    'Date': date,
                    'Price': price,
                    'Open': price,  
                    'High': price * 1.005,
                    'Low': price * 0.995,
                    'Vol.': volume
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values("Date").reset_index(drop=True)
            
            return df
            
        except requests.RequestException as e:
            raise Exception(f"Network error while fetching data: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to fetch real-time data: {str(e)}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from raw price data"""
        try:
            df_copy = df.copy()
            
            # Ensure we have required columns
            required_cols = ['Price', 'Open', 'High', 'Low', 'Vol.']
            missing_cols = [col for col in required_cols if col not in df_copy.columns]
            if missing_cols:
                raise Exception(f"Missing required columns: {missing_cols}")
            
            # Lag features
            df_copy['Price_lag1'] = df_copy['Price'].shift(1)
            df_copy['Price_lag2'] = df_copy['Price'].shift(2)
            df_copy['Vol_lag1'] = df_copy['Vol.'].shift(1)
            df_copy['Open_lag1'] = df_copy['Open'].shift(1)
            df_copy['High_lag1'] = df_copy['High'].shift(1)
            df_copy['Low_lag1'] = df_copy['Low'].shift(1)
            
            # Moving averages
            df_copy['MA3'] = df_copy['Price'].rolling(window=3).mean().shift(1)
            df_copy['MA5'] = df_copy['Price'].rolling(window=5).mean().shift(1)
            
            # Drop NaN values
            df_copy = df_copy.dropna().reset_index(drop=True)
            
            return df_copy
            
        except Exception as e:
            raise Exception(f"Failed to create features: {str(e)}")
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML training"""
        try:
            # Check if all required features exist
            missing_features = [feat for feat in self.features if feat not in df.columns]
            if missing_features:
                raise Exception(f"Missing required features: {missing_features}")
            
            X = df[self.features].copy()
            y = df["Price"].copy()
            
            # Final check for NaN values
            if X.isnull().any().any():
                nan_mask = X.isnull().any(axis=1)
                X = X[~nan_mask]
                y = y[~nan_mask]
            
            if len(X) < 10:
                raise Exception("Insufficient data after cleaning: need at least 10 samples")
            
            return X, y
            
        except Exception as e:
            raise Exception(f"Failed to prepare training data: {str(e)}")
    
    def get_latest_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract latest feature values for prediction"""
        try:
            if len(df) == 0:
                raise Exception("DataFrame is empty")
            
            latest_row = df.iloc[-1]
            
            # Extract features in the correct order
            latest_features = {}
            for feature in self.features:
                if feature not in latest_row:
                    raise Exception(f"Feature {feature} not found in data")
                latest_features[feature] = float(latest_row[feature])
            
            return latest_features
            
        except Exception as e:
            raise Exception(f"Failed to extract latest features: {str(e)}")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive data information"""
        try:
            if df.empty:
                return {
                    "total_rows": 0,
                    "date_range": {"start": None, "end": None},
                    "latest_price": None,
                    "price_stats": {},
                    "columns": []
                }
            
            date_range = {
                "start": df['Date'].min().strftime("%Y-%m-%d") if 'Date' in df.columns else None,
                "end": df['Date'].max().strftime("%Y-%m-%d") if 'Date' in df.columns else None
            }
            
            price_stats = {
                "min": float(df['Price'].min()),
                "max": float(df['Price'].max()),
                "mean": float(df['Price'].mean()),
                "std": float(df['Price'].std())
            } if 'Price' in df.columns else {}
            
            return {
                "total_rows": len(df),
                "date_range": date_range,
                "latest_price": float(df['Price'].iloc[-1]) if 'Price' in df.columns and len(df) > 0 else None,
                "price_stats": price_stats,
                "columns": df.columns.tolist()
            }
            
        except Exception as e:
            return {"error": f"Failed to get data info: {str(e)}"}
    
    async def get_current_price(self) -> Dict:
        """Get current Ethereum price from CoinGecko API"""
        try:
            url = f"{settings.COINGECKO_API_URL}/simple/price"
            params = {
                'ids': 'ethereum',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'ethereum' not in data:
                raise Exception("Ethereum data not found in API response")
            
            eth_data = data['ethereum']
            
            return {
                "current_price": eth_data.get('usd', 0),
                "price_change_24h": eth_data.get('usd_24h_change', 0),
                "volume_24h": eth_data.get('usd_24h_vol', 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch current price: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing price data: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality and completeness"""
        try:
            validation_results = {
                "is_valid": True,
                "warnings": [],
                "errors": []
            }
            
            # Check for required columns
            required_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results["errors"].append(f"Missing columns: {missing_columns}")
                validation_results["is_valid"] = False
            
            # Check for sufficient data
            if len(df) < 10:
                validation_results["errors"].append(f"Insufficient data: {len(df)} rows (need at least 10)")
                validation_results["is_valid"] = False
            
            # Check for missing values in critical columns
            if 'Price' in df.columns:
                null_prices = df['Price'].isnull().sum()
                if null_prices > 0:
                    validation_results["warnings"].append(f"Found {null_prices} missing price values")
            
            # Check for unrealistic price values
            if 'Price' in df.columns and len(df) > 0:
                min_price = df['Price'].min()
                max_price = df['Price'].max()
                if min_price <= 0:
                    validation_results["errors"].append("Found non-positive price values")
                    validation_results["is_valid"] = False
                if max_price > 100000:  # Sanity check for ETH price
                    validation_results["warnings"].append(f"Unusually high price detected: ${max_price:,.2f}")
            
            return validation_results
            
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": []
            }