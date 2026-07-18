import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core.util.logger import ILogger

sns.set_theme(style="whitegrid")

class EvaluateModelsUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Avoid deviode by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    def plot_forecast_comparison(self, test_dates: pd.Series, y_true: np.ndarray, model_predictions: dict, save_dir: str = "plots") -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info("Plotting forecast comparison")
        plt.figure(figsize=(12, 6))
        
        plt.plot(test_dates, y_true, label='Actual Price', color='black', linewidth=1.5)
        
        colors = {'ARIMA': 'royalblue', 'XGBoost': 'forestgreen', 'LSTM': 'crimson'}
        for name, preds in model_predictions.items():
            plt.plot(test_dates, preds, label=f'{name} Forecast', color=colors.get(name, 'gray'), linestyle='--', linewidth=1.2)
            
        plt.title('Bitcoin Price Forecasting Model Comparison (Test Set)', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison_forecast.png'), dpi=150)
        plt.close()