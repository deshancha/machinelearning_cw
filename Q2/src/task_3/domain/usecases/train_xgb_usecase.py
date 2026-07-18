import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from core.util.logger import ILogger

class TrainXgbUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def prepare_ml_data(self, df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
        # Drop non feature columns
        exclude = ['Date', target_col, 'Returns', 'Volume', 'Open', 'High', 'Low', 'Adj Close']
        feature_cols = [c for c in df.columns if c not in exclude]
        return df[feature_cols], df[target_col]

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        # Grid search XGBoost hyperparameters and select lowest validation RMSE.
        self.logger.info("Tuning XGBoost parameters")
        best_rmse = float("inf")
        best_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
        
        # Grid parameters
        learning_rates = [0.03, 0.1]
        max_depths = [3, 5]
        n_estimators = [100, 200]
        
        for lr in learning_rates:
            for depth in max_depths:
                for nest in n_estimators:
                    model = xgb.XGBRegressor(
                        n_estimators=nest,
                        max_depth=depth,
                        learning_rate=lr,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    val_preds = model.predict(X_val)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                    
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        best_params = {
                            'n_estimators': nest,
                            'max_depth': depth,
                            'learning_rate': lr
                        }
                        
        self.logger.info(f"Best XGBoost: {best_params}")
        return best_params

    def train_and_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, best_params: dict, model_path: str = None) -> np.ndarray:
        self.logger.info("Training final XGBoost model")
        model = xgb.XGBRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model if path provided
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            
        return model.predict(X_test)
