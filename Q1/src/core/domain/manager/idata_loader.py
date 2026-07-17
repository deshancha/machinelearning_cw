from abc import ABC, abstractmethod
import pandas as pd

class IDataLoader(ABC):
    @abstractmethod
    def fetch_from_uci(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_raw_data(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def load_processed_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_processed_data(self, df: pd.DataFrame) -> None:
        pass
