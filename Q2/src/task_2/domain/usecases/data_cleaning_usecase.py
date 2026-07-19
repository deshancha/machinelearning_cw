# Author: Chamika Deshan

import pandas as pd
import numpy as np
from core.util.logger import ILogger

class DataCleaningUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def execute(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        self.logger.info("Start data cleaning")
        
        if df is None or df.empty:
            self.logger.error("DataFrame is empty")
            return None

        df = df.copy()

        # Drop source
        if 'source' in df.columns:
            df = df.drop(columns=['source'])

        # Rename columns
        rename_map = {
            'timestamp_utc': 'Date',
            'open_price': 'Open',
            'high_price': 'High',
            'low_price': 'Low',
            'close_price': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=rename_map)
        
        # Handle Missing Values
        initial_missing = df.isnull().sum().sum()
        if initial_missing > 0:
            df = df.ffill().bfill()
        else:
            self.logger.info("No missing found")

        # Convert Date 
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # Convert numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # handle negative [price,volume] anomalies
        numeric_df_cols = df.select_dtypes(include=[np.number]).columns
        negative_counts = (df[numeric_df_cols] < 0).sum()
        if negative_counts.sum() > 0:
            for col in numeric_df_cols:
                df.loc[df[col] < 0, col] = np.nan
            df = df.ffill().bfill()
            self.logger.info("Negative anomalies handled")
        else:
            self.logger.info("No negative found")

        # Compute percentage returns for volatility analysis
        df['Returns'] = df['Close'].pct_change() * 100
        df['Returns'] = df['Returns'].fillna(0.0)

        
        cleaned_missing = df.isnull().sum().sum()
        self.logger.info(f"cleaning done. missing: {cleaned_missing}")
        
        return df, 'Close'
