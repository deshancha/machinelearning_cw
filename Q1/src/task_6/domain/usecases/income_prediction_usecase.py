import os
import joblib
import pandas as pd
import dvc.api
from typing import List
from core.util.logger import ILogger
from task_6.domain.model.models import PredictionInput, PredictionResponse

class IncomePredictionUseCase:
    def __init__(self, model_dir: str, logger: ILogger):
        self.logger = logger
        self.model_dir = model_dir
        self.model_path = "Q1/outputs/models/xgboost_model.joblib"
        self.model = None
        
        self._load_model()

    def _load_model(self):
        self.logger.info(f"IncomePredictionUseCase loading model via DVC programmatically from: {self.model_path}")
        try:
            with dvc.api.open(
                path=self.model_path,
                mode='rb'
            ) as model_file:
                self.model = joblib.load(model_file)
            self.logger.info("Loaded XGBoost model via DVC OK")
        except Exception as e:
            self.logger.error(f"Failed to load model via DVC: {str(e)}")
            raise e

    def execute(self, inputs: List[PredictionInput]) -> List[PredictionResponse]:
        self.logger.info(f"Exec IncomePredictionUseCase for {len(inputs)} records")
        
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Inputs to raw column conversion with Pydantic
        raw_data = [item.model_dump(by_alias=True) for item in inputs]
        df = pd.DataFrame(raw_data)

        # Feature Eng
        df['capital_net'] = df['capital-gain'] - df['capital-loss']
        
        df['has_capital'] = ((df['capital-gain'] > 0) | (df['capital-loss'] > 0)).astype(int)
        
        df['age_group'] = pd.cut(df['age'],
                                 bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=['Young', 'Early-Career', 'Mid-Career',
                                         'Senior', 'Pre-Retire', 'Retired'])
        
        df['hours_category'] = pd.cut(df['hours-per-week'],
                                      bins=[0, 34, 40, 100],
                                      labels=['Part-Time', 'Full-Time', 'Overtime'])
        
        edu_map = {
            'Preschool': 'Low', '1st-4th': 'Low', '5th-6th': 'Low', '7th-8th': 'Low',
            '9th': 'Low', '10th': 'Low', '11th': 'Low', '12th': 'Low',
            'HS-grad': 'High-School',
            'Some-college': 'Some-College', 'Assoc-voc': 'Some-College', 'Assoc-acdm': 'Some-College',
            'Bachelors': 'Bachelors',
            'Masters': 'Advanced', 'Prof-school': 'Advanced', 'Doctorate': 'Advanced'
        }
        df['edu_group'] = df['education'].map(edu_map)

        # Matching the column order as trained model
        expected_order = [
            'age', 'workclass', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain',
            'capital-loss', 'hours-per-week', 'native-country', 'capital_net',
            'has_capital', 'age_group', 'hours_category', 'edu_group'
        ]
        df = df[expected_order]

        # We can predict now
        preds = self.model.predict(df)
        proba = self.model.predict_proba(df)[:, 1]

        # Pydantic Format prediction response
        results = []
        for pred, prob in zip(preds, proba):
            results.append(PredictionResponse(
                prediction=int(pred),
                label=">50K" if pred == 1 else "<=50K",
                probability=float(prob)
            ))
            
        self.logger.info("Batch inference completed successfully.")
        return results
