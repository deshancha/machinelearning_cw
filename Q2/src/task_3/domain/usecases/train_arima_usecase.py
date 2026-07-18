import os
import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from core.util.logger import ILogger

class TrainArimaUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def tune(self, train_series: pd.Series, p_values=[0, 1, 2], d_values=[1], q_values=[0, 1, 2]) -> tuple:
        """
        Grid search ARIMA -> lowest AIC.
        """
        self.logger.info("Tuning ARIMA parameters")
        best_aic = float("inf")
        best_order = (1, 1, 1) # Default baseline order
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        # Fit model on training series
                        model = SARIMAX(train_series, order=(p, d, q), enforce_stationarity=False)
                        fit_res = model.fit(disp=False)
                        
                        # Select order with the lowest Akaike Information Criterion (AIC)
                        if fit_res.aic < best_aic:
                            best_aic = fit_res.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        self.logger.info(f"Best ARIMA: {best_order}")
        return best_order

    def evaluate(self, train_series: pd.Series, test_series: pd.Series, order: tuple, model_path: str = None) -> pd.Series:
        self.logger.info(f"Fitting ARIMA{order}")
        model = SARIMAX(train_series, order=order, enforce_stationarity=False)
        model_fit = model.fit(disp=False)
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model_fit, model_path)
            
        # Concatenate train and test series to get rolling forecasts
        full_series = pd.concat([train_series, test_series])
        res = model_fit.apply(full_series)
        
        # Extract predictions corresponding to the test set period
        return res.fittedvalues.iloc[-len(test_series):]
