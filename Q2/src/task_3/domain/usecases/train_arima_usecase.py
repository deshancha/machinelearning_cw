import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from core.util.logger import ILogger
import os
import joblib

class TrainArimaUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def tune(self, train_series: pd.Series, p_values: list = [0, 1, 2], d_values: list = [0, 1], q_values: list = [0, 1, 2]) -> tuple:
        self.logger.info("Tuning ARIMA params")
        best_aic = float("inf")
        best_order = None
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        # ARIMA(p,d,q) baseline (no seasonal order 365 since daily data is too slow to fit)
                        model = SARIMAX(train_series, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                        results = model.fit(disp=False)
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except:
                        continue
                        
        self.logger.info(f"Best ARIMA Order: {best_order} with AIC: {best_aic:.2f}")
        return best_order

    def evaluate(self, train_series: pd.Series, test_series: pd.Series, order: tuple, model_path: str = None) -> pd.Series:
        self.logger.info(f"Fitting ARIMA{order} on training data")
        model = SARIMAX(train_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model_fit, model_path)
            self.logger.info(f"Saved ARIMA model to {model_path}")
        
        # Apply  params to full series -> get better forecasting
        full_series = pd.concat([train_series, test_series])
        res = model_fit.apply(full_series)
        
        self.logger.info("Predicting on test set")
        predictions = res.fittedvalues.iloc[-len(test_series):]
        return predictions
