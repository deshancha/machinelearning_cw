import pandas as pd
from core.util.logger import ILogger

class FeatureEngineeringUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def execute(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        self.logger.info("Starting feature engineering...")
        df = df.copy()
        
        # 1. Calendar Features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # 2. Lag Features
        lags = [1, 2, 3, 5, 7, 14, 30]
        for lag in lags:
            df[f'Lag_{lag}'] = df[close_col].shift(lag)
            
            
        # Drop rows containing NaNs
        df_cleaned = df.dropna().reset_index(drop=True)
        
        self.logger.info(f"Orignal shape {df.shape}, cleaned shape {df_cleaned.shape}.")
        
        return df_cleaned
