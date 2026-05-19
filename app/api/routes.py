from pathlib import Path
import math

import numpy as np

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.config import settings
from app.core.model import ModelStore
from app.schemas.predict import (
    PredictRequest,
    PredictResponse,
    PredictProbaResponse,
    PredictFeaturesRequest,
    ThresholdInfo,
    PredictionItem,
)
from app.services.predict import predict_records, predict_parquet_bytes, predict_feature_records

router = APIRouter()


def _sanitize_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _build_threshold_info() -> ThresholdInfo:
    bundle = ModelStore.get_bundle()
    metrics = bundle.metrics or {}
    
    return ThresholdInfo(
        umbral_global=bundle.thresholds.get("umbral", settings.threshold_global),
        test_year=bundle.test_year if bundle.test_year is not None else settings.test_year,
        test_households=metrics.get("n_hogares") or metrics.get("hogares") or settings.test_households,
        tasa_real_pct=metrics.get("tasa_real_pct") or metrics.get("tasa_real") or settings.test_real_rate_pct,
        tasa_predicha_pct=(
            metrics.get("tasa_predicha_pct")
            or metrics.get("tasa_predicha")
            or settings.test_pred_rate_pct
        ),
    )


def _build_prediction_items(predictions: list[float], threshold: float) -> list[PredictionItem]:
    return [PredictionItem(prediction=value, umbral_global=threshold) for value in predictions]


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": ModelStore.is_loaded(),
        "threshold_info": _build_threshold_info() if ModelStore.is_loaded() else None,
    }


@router.get("/model/info")
def model_info():
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = ModelStore.get_model()
    return {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "supports_predict_proba": hasattr(model, "predict_proba"),
        "supports_predict": hasattr(model, "predict"),
        "feature_columns": settings.feature_columns,
        "threshold_info": _build_threshold_info(),
    }


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions, _ = predict_records(payload.records)
    threshold_info = _build_threshold_info()
    return PredictResponse(
        predictions=_build_prediction_items(predictions, threshold_info.umbral_global),
        threshold_info=threshold_info,
    )


@router.post("/predict/proba", response_model=PredictProbaResponse)
def predict_proba(payload: PredictRequest):
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    _, probabilities = predict_records(payload.records)
    return PredictProbaResponse(
        probabilities=probabilities,
        threshold_info=_build_threshold_info(),
    )


@router.post("/predict/features", response_model=PredictResponse)
def predict_features(payload: PredictFeaturesRequest):
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions, _ = predict_feature_records(payload.records, settings.feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    threshold_info = _build_threshold_info()
    return PredictResponse(
        predictions=_build_prediction_items(predictions, threshold_info.umbral_global),
        threshold_info=threshold_info,
    )


@router.post("/predict/features/proba", response_model=PredictProbaResponse)
def predict_features_proba(payload: PredictFeaturesRequest):
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        _, probabilities = predict_feature_records(payload.records, settings.feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictProbaResponse(
        probabilities=probabilities,
        threshold_info=_build_threshold_info(),
    )


@router.post("/predict/parquet", response_model=PredictResponse)
async def predict_parquet(file: UploadFile = File(...)):
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = await file.read()
    try:
        predictions, _ = predict_parquet_bytes(data, settings.feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    threshold_info = _build_threshold_info()
    return PredictResponse(
        predictions=_build_prediction_items(predictions, threshold_info.umbral_global),
        threshold_info=threshold_info,
    )


@router.post("/predict/parquet/proba", response_model=PredictProbaResponse)
async def predict_parquet_proba(file: UploadFile = File(...)):
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = await file.read()
    try:
        _, probabilities = predict_parquet_bytes(data, settings.feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictProbaResponse(
        probabilities=probabilities,
        threshold_info=_build_threshold_info(),
    )


@router.get("/predict/parquet/default", response_model=PredictResponse)
def predict_parquet_default():
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    path = Path("datasets/panel_procesado.parquet")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    data = path.read_bytes()
    try:
        predictions, _ = predict_parquet_bytes(data, settings.feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    threshold_info = _build_threshold_info()
    return PredictResponse(
        predictions=_build_prediction_items(predictions, threshold_info.umbral_global),
        threshold_info=threshold_info,
    )


@router.get("/predict/parquet/default/proba", response_model=PredictProbaResponse)
def predict_parquet_default_proba():
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    path = Path("datasets/panel_procesado.parquet")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    data = path.read_bytes()
    try:
        _, probabilities = predict_parquet_bytes(data, settings.feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictProbaResponse(
        probabilities=probabilities,
        threshold_info=_build_threshold_info(),
    )