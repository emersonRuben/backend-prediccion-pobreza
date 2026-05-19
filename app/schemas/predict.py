from typing import Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    records: list[list[float]] = Field(..., min_length=1)


class PredictFeaturesRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class ThresholdInfo(BaseModel):
    umbral_global: float
    test_year: int | None = None
    test_households: int | None = None
    tasa_real_pct: float | None = None
    tasa_predicha_pct: float | None = None


class PredictionItem(BaseModel):
    prediction: float
    umbral_global: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    threshold_info: ThresholdInfo


class PredictProbaResponse(BaseModel):
    probabilities: list[list[float]] | None = None
    threshold_info: ThresholdInfo
