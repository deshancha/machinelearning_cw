from typing import List
from pydantic import BaseModel, Field, ConfigDict

class PredictionInput(BaseModel):
    age: int = Field(..., json_schema_extra={"example": 39})
    workclass: str = Field(..., json_schema_extra={"example": "State-gov"})
    education: str = Field(..., json_schema_extra={"example": "Bachelors"})
    education_num: int = Field(..., alias="education-num", json_schema_extra={"example": 13})
    marital_status: str = Field(..., alias="marital-status", json_schema_extra={"example": "Never-married"})
    occupation: str = Field(..., json_schema_extra={"example": "Adm-clerical"})
    relationship: str = Field(..., json_schema_extra={"example": "Not-in-family"})
    race: str = Field(..., json_schema_extra={"example": "White"})
    sex: str = Field(..., json_schema_extra={"example": "Male"})
    capital_gain: int = Field(..., alias="capital-gain", json_schema_extra={"example": 2174})
    capital_loss: int = Field(..., alias="capital-loss", json_schema_extra={"example": 0})
    hours_per_week: int = Field(..., alias="hours-per-week", json_schema_extra={"example": 40})
    native_country: str = Field(..., alias="native-country", json_schema_extra={"example": "United-States"})

    model_config = ConfigDict(
        populate_by_name=True
    )

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for <=50K, 1 for >50K")
    label: str = Field(..., description="Income category label (<=50K or >50K)")
    probability: float = Field(..., description="Probability of income being >50K")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
