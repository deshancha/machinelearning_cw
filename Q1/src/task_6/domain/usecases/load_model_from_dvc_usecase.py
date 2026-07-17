import dvc.api
import joblib
from core.util.logger import ILogger

class LoadModelFromDvcUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def execute(self, model_path: str):
        try:
            with dvc.api.open(
                path=model_path,
                mode='rb'
            ) as model_file:
                model = joblib.load(model_file)
            self.logger.info("Loaded XGBoost model via DVC")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model via DVC: {str(e)}")
            raise e
