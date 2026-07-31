import os
import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from core.util.logger import ILogger

class TrainLstmUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def train_and_predict(self, train_series: pd.Series, test_series: pd.Series, epochs: int = 50, model_path: str = None) -> pd.Series:
        self.logger.info("pandas to Darts TimeSeries")
        
        train_ts = TimeSeries.from_series(train_series)
        
        scaler = Scaler()
        train_scaled = scaler.fit_transform(train_ts)
        
        self.logger.info("Training RNNM (LSTM)")
        # RNN with LSTM
        model = RNNModel(
            model="LSTM",
            n_epochs=epochs
        )
        
        model.fit(train_scaled)
        
        self.logger.info("forecasts...")
       
        # Forecast
        pred_scaled = model.predict(n=len(test_series), series=train_scaled)
        
        # back to original scale
        pred_ts = scaler.inverse_transform(pred_scaled)
        preds_original = pd.Series(pred_ts.values().flatten(), index=test_series.index)
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            self.logger.info(f"Saved LSTM to {model_path}")
            
        return preds_original
