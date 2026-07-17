import numpy as np
import pandas as pd
from core.domain.manager.idata_loader import IDataLoader
from core.util.logger import ILogger

class DataCleaningUseCase:
    def __init__(self, data_loader: IDataLoader, logger: ILogger):
        self.data_loader = data_loader
        self.logger = logger

    def execute(self) -> pd.DataFrame:
        df = self.data_loader.load_raw_data()
        
        self.logger.info(f"Initial shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        
        # Replace ? with NaN
        df.replace('?', np.nan, inplace=True)
        
        # Drop rows with missing values
        rows_before_drop = len(df)
        df.dropna(inplace=True)
        dropped_missing = rows_before_drop - len(df)
        self.logger.info(f"Dropped {dropped_missing:,} rows with missing values")
        
        # Remove duplicate records
        dup_before = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        self.logger.info(f"Removed {dup_before:,} duplicate rows")
        
        # Clean target class values and make a binary income column
        df['income_binary'] = (df['income'].astype(str).str.strip() == '>50K').astype(int)
        
        # Calculate numerical correlations vs target before dropping fnlwgt
        numerical_cols = ['age', 'fnlwgt', 'education-num', 'hours-per-week', 'income_binary']
        available_cols = [c for c in numerical_cols if c in df.columns]
        if 'income_binary' in available_cols:
            correlations = df[available_cols].corr()['income_binary']
            self.logger.info(f"Correlations with target income:\n{correlations.to_string()}")
            
        # Drop fnlwgt column
        if 'fnlwgt' in df.columns:
            df.drop(columns=['fnlwgt'], inplace=True, errors='ignore')
            self.logger.info("Dropped column 'fnlwgt', near zero correlation with target")
            
        self.logger.info(f"Output shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df
