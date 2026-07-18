import os
import pandas as pd
import numpy as np
from core.domain.manager.idata_loader import IDataLoader
from core.util.logger import ILogger
from core.domain.usecases.collect_market_data_usecase import CollectMarketDataUseCase
from task_2.domain.usecases.data_cleaning_usecase import DataCleaningUseCase

class YahooFinanceLoaderImp(IDataLoader):
    def __init__(self, logger: ILogger, collect_usecase: CollectMarketDataUseCase, cleaning_usecase: DataCleaningUseCase, data_dir: str = None):
        self.logger = logger
        self.collect_usecase = collect_usecase
        self.cleaning_usecase = cleaning_usecase
        
        if data_dir is None:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data"))
        self.data_dir = data_dir
        self.raw_path = os.path.join(self.data_dir, "btc_raw.csv")
        self.cleaned_path = os.path.join(self.data_dir, "btc_cleaned.csv")
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self, ticker: str = "BTC-USD", start: str = "2020-01-01", end: str = "2026-06-30") -> pd.DataFrame:
        # Fetch using collect_usecase (which returns List[AssetData])
        assets = self.collect_usecase.fetch_daily(ticker, start, end)
        
        if not assets:
            self.logger.error(f"No data retrieved for ticker {ticker}.")
            return None
            
        # Convert List[AssetData] to DataFrame with original raw fields
        data_list = []
        for asset in assets:
            data_list.append({
                'timestamp_utc': asset.timestamp_utc,
                'open_price': asset.open_price,
                'high_price': asset.high_price,
                'low_price': asset.low_price,
                'close_price': asset.close_price,
                'volume': asset.volume,
                'source': asset.source
            })
        df = pd.DataFrame(data_list)
        
        self.logger.info(f"Loaded {len(df)} rows from CollectMarketDataUseCase.")
        self.save_raw_data(df)
        return df

    def clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        # Delegate cleaning to DataCleaningUseCase
        cleaned_df, close_col = self.cleaning_usecase.execute(df)
        self.save_cleaned_data(cleaned_df)
        return cleaned_df, close_col

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)
        
        return train_df, val_df, test_df

    def load_raw_data(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            self.logger.error(f"Raw data file not found at {self.raw_path}!")
            return None
        df = pd.read_csv(self.raw_path)
        self.logger.info(f"Loaded raw data: {df.shape[0]} rows")
        return df

    def save_raw_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.raw_path, index=False)
        self.logger.info(f"Raw data saved to {self.raw_path}")

    def load_cleaned_data(self) -> pd.DataFrame:
        if not os.path.exists(self.cleaned_path):
            self.logger.error(f"Cleaned data file not found at {self.cleaned_path}!")
            return None
        df = pd.read_csv(self.cleaned_path)
        df['Date'] = pd.to_datetime(df['Date'])
        self.logger.info(f"Loaded cleaned data: {df.shape[0]} rows")
        return df

    def save_cleaned_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.cleaned_path, index=False)
        self.logger.info(f"Cleaned data saved to {self.cleaned_path}")
