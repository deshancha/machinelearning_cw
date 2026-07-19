import pandas as pd
from statsmodels.tsa.stattools import adfuller
from core.util.logger import ILogger

class StationarityTestUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def execute(self, df: pd.DataFrame, col_name: str) -> tuple[dict, bool]:
        self.logger.info(f"Performing ADF test :'{col_name}'...")
        series = df[col_name].dropna()
        result = adfuller(series, autolag='AIC')
        
        report = {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Lags Used': result[2],
            'Number of Observations': result[3],
            'Critical Values': result[4]
        }
        
        # Significance level is 5%
        is_stationary = result[1] <= 0.05
        
        self.logger.info(f"ADF Statistic : {result[0]:.6f}")
        self.logger.info(f"p-value : {result[1]:.6e}")
        self.logger.info(f"Stationarity Result : {'Stationary' if is_stationary else 'Non-Stationary'} (5% Significance level)")
        
        return report, is_stationary
