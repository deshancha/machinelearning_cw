import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from core.util.logger import ILogger
import os
import joblib

class TrainXgbUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def prepare_ml_data(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        feature_cols = [c for c in df.columns if c not in ['Date', target_col, 'Returns', 'Volume', 'Open', 'High', 'Low', 'Adj Close']]
        X = df[feature_cols]
        y = df[target_col]
        return X, y

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        self.logger.info("Tuning XGBoost hyperparams")
        best_val_rmse = float("inf")
        best_params = None
        
        # Grid search parameters
        learning_rates = [0.03, 0.1]
        max_depths = [3, 5, 7]
        n_estimators = [100, 200]
        
        for lr in learning_rates:
            for depth in max_depths:
                for nest in n_estimators:
                    model = xgb.XGBRegressor(
                        n_estimators=nest,
                        max_depth=depth,
                        learning_rate=lr,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    val_preds = model.predict(X_val)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                    
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_params = {
                            'n_estimators': nest,
                            'max_depth': depth,
                            'learning_rate': lr
                        }
                        
        self.logger.info(f"Best XGBoost Params: {best_params} with Validation RMSE: {best_val_rmse:.2f}")
        return best_params

    def train_and_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, best_params: dict, model_path: str = None) -> np.ndarray:
        self.logger.info("Training XGBoost model")
        model = xgb.XGBRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            self.logger.info(f"Saved XGBoost model to {model_path}")
            
        # predict
        self.logger.info("Predicting on test set")
        preds = model.predict(X_test)
        return preds
