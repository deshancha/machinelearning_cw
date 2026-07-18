# Author: Chamika Deshan
# Adapted for Q2

from abc import ABC, abstractmethod
from typing import List
from core.domain.model.asset_data import AssetData

class IMarketDataClient(ABC):
    
    @abstractmethod
    def fetch_daily(self, ticker: str, start_date: str, end_date: str) -> List[AssetData]:
        pass
