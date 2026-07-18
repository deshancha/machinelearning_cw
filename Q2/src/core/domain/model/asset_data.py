# Author: Chamika Deshan

from pydantic import BaseModel
from typing import Optional

class AssetData(BaseModel):
    """
    Market Data Model Instance
    """
    ticker: str
    timestamp_utc: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    source: Optional[str] = "unknown"
