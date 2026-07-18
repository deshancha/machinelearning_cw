# Author: Chamika Deshan

from typing import List
from core.domain.manager.imarket_data_client import IMarketDataClient
from core.domain.model.asset_data import AssetData

class CollectMarketDataUseCase:
    def __init__(self, market_client: IMarketDataClient):
        self.market_client = market_client
        
    def fetch_daily(self, ticker: str, start_date: str, end_date: str) -> List[AssetData]:
        return self.market_client.fetch_daily(ticker, start_date, end_date)
