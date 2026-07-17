import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from core.domain.manager.idata_loader import IDataLoader
from core.util.logger import ILogger

class CsvLoaderImp(IDataLoader):
    def __init__(self, data_dir: str, logger: ILogger):
        self.data_dir = data_dir
        self.logger = logger
        self.raw_path = os.path.join(self.data_dir, "adult_income_raw.csv")
        self.processed_path = os.path.join(self.data_dir, "adult_income_processed.csv")
        
        # Ensure directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_from_uci(self) -> pd.DataFrame:
        adult = fetch_ucirepo(id=2)
        
        X = adult.data.features
        y = adult.data.targets
        
        self.logger.info(f"Fetched dataset: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Concatenate features and targets
        df = pd.concat([X, y], axis=1)
        
        # Clean the target values (strip spaces and trailing dots)
        target_col = y.columns[0]
        df[target_col] = df[target_col].astype(str).str.strip().str.rstrip('.')
        
        return df

    def load_raw_data(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            self.logger.error(f"Raw data file not found at {self.raw_path}!")
            raise FileNotFoundError(f"Raw data file not found at {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        self.logger.info(f"Loaded raw data: {df.shape[0]} rows")
        return df

    def save_raw_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.raw_path, index=False)
        self.logger.info("Raw data saved successfully")

    def load_processed_data(self) -> pd.DataFrame:
        if not os.path.exists(self.processed_path):
            self.logger.error(f"Processed data file not found at {self.processed_path}!")
            raise FileNotFoundError(f"Processed data file not found at {self.processed_path}")
        df = pd.read_csv(self.processed_path)
        self.logger.info(f"Loaded processed data: {df.shape[0]} rows")
        return df

    def save_processed_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.processed_path, index=False)
        self.logger.info(f"Processed data saved successfully {self.processed_path}")
