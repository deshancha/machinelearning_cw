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
        
        # Determine closing column
        close_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
        if not close_col:
            # Fallback search
            for col in df.columns:
                if 'Close' in col or 'close' in col:
                    close_col = col
                    break
        if not close_col:
            self.logger.error("Could not find a closing price column.")
            return None
            
        self.logger.info(f"Using column '{close_col}' as the closing price.")

        # Handle Missing Values
        initial_missing = df.isnull().sum().sum()
        if initial_missing > 0:
            self.logger.info(f"Handling {initial_missing} missing values")
            df = df.ffill().bfill()
            self.logger.info("Missing values imputed using ffill/bfill.")
        else:
            self.logger.info("No missing values found.")

        # Convert Date column type
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for negative price/volume anomalies
        numeric_df_cols = df.select_dtypes(include=[np.number]).columns
        negative_counts = (df[numeric_df_cols] < 0).sum()
        if negative_counts.sum() > 0:
            self.logger.warn("Negative values found in numeric columns:")
            self.logger.warn(str(negative_counts[negative_counts > 0]))
            for col in numeric_df_cols:
                df.loc[df[col] < 0, col] = np.nan
            df = df.ffill().bfill()
            self.logger.info("Negative value anomalies resolved.")

        # Compute simple percentage returns for volatility analysis (Percentage Change)
        df['Returns'] = df[close_col].pct_change()
        df['Returns'] = df['Returns'].fillna(0.0)

        cleaned_missing = df.isnull().sum().sum()
        self.logger.info(f"Data cleaning complete. Missing: {cleaned_missing}")
        
        return df, close_col
