import uvicorn
from typing import List
from fastapi import FastAPI, HTTPException
from core.util.logger import ILogger
from task_6.domain.manager.iapi_server import IApiServer
from task_6.domain.usecases.income_prediction_usecase import IncomePredictionUseCase
from task_6.domain.model.models import PredictionInput, PredictionResponse, BatchPredictionResponse

class ApiServerImp(IApiServer):
    def __init__(self, usecase: IncomePredictionUseCase, logger: ILogger):
        self.usecase = usecase
        self.logger = logger
        
        self.api = FastAPI(
            title="Adult Income Prediction API",
            description="Api service for real time and batch income predictions using XGBoost with Clean architecture",
            version="1.0.0"
        )
        
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.api.post("/predict", response_model=PredictionResponse)
        def predict(input_data: PredictionInput):
            self.logger.info("Single prediction request received")
            try:
                results = self.usecase.execute([input_data])
                return results[0]
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

        @self.api.post("/predict_batch", response_model=BatchPredictionResponse)
        def predict_batch(input_data: List[PredictionInput]):
            self.logger.info(f"Batch prediction request received for {len(input_data)} records.")
            try:
                results = self.usecase.execute(input_data)
                return BatchPredictionResponse(predictions=results)
            except Exception as e:
                self.logger.error(f"Batch prediction error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

    def start(self, host: str, port: int) -> None:
        self.logger.info(f"Start api server -> {host}:{port}...")
        uvicorn.run(self.api, host=host, port=port, reload=False)
