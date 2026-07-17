import pandas as pd
from core.domain.manager.idata_loader import IDataLoader
from core.util.logger import ILogger

class FetchRawDataUseCase:
    def __init__(self, data_loader: IDataLoader, logger: ILogger):
        self.data_loader = data_loader
        self.logger = logger

    def execute(self) -> pd.DataFrame:
        df = self.data_loader.fetch_from_uci()
        self.data_loader.save_raw_data(df)
        self.logger.info("FetchRawDataUseCase execution complete")
        return df
