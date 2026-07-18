from abc import ABC, abstractmethod
import pandas as pd

class IDataLoader(ABC):
    @abstractmethod
    def load_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        pass

    @abstractmethod
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_raw_data(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def load_cleaned_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_cleaned_data(self, df: pd.DataFrame) -> None:
        pass
