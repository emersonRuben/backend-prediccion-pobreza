from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core.model import ModelStore
from app.core.panel import PanelStore
from app.schemas.dashboard import (
    AniosResponse,
    HealthResponse,
    ModeloInfoResponse,
    PredictObservacionesRequest,
    PredictObservacionesResponse,
    RegionalResponse,
    ResumenAnualResponse,
    TendenciaAreaResponse,
    TendenciaNacionalResponse,
    TendenciaPoblacionResponse,
    TendenciaRegionalResponse,
)
from app.services.dashboard import (
    area_trend,
    get_years,
    national_trend,
    population_trend,
    predict_observations,
    regional_rates,
    regional_trend,
    summarize_year,
)

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "modelo_cargado": ModelStore.is_loaded(),
        "panel_cargado": PanelStore.is_loaded(),
        "n_registros_panel": PanelStore.rows(),
    }


@router.get("/modelo/info", response_model=ModeloInfoResponse)
def modelo_info():
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    bundle = ModelStore.get_bundle()
    return {
        "anio_test": bundle.test_year,
        "umbral": bundle.thresholds.get("umbral", 0.5),
        "umbral_f1": bundle.thresholds.get("umbral_f1", 0.5),
        "umbral_r80": bundle.thresholds.get("umbral_r80", 0.5),
        "n_features": len(bundle.features),
        "metricas_test": bundle.metrics,
    }


@router.get("/anios", response_model=AniosResponse)
def anios_disponibles():
    try:
        years = get_years()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"anios": years}


@router.get("/resumen/{anio}", response_model=ResumenAnualResponse)
def resumen_anual(anio: int):
    try:
        return summarize_year(anio)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/regional/{anio}", response_model=RegionalResponse)
def pobreza_regional(anio: int, top_n: int | None = Query(default=25, ge=1)):
    try:
        return regional_rates(anio, top_n)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/tendencia/nacional", response_model=TendenciaNacionalResponse)
def tendencia_nacional():
    try:
        return national_trend()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/tendencia/area", response_model=TendenciaAreaResponse)
def tendencia_area():
    try:
        return area_trend()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/tendencia/poblacion", response_model=TendenciaPoblacionResponse)
def tendencia_poblacion():
    try:
        return population_trend()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/tendencia/regional/{departamento_id}", response_model=TendenciaRegionalResponse)
def tendencia_regional(departamento_id: int):
    try:
        return regional_trend(departamento_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictObservacionesResponse)
def predict_endpoint(payload: PredictObservacionesRequest):
    try:
        return predict_observations(payload.observaciones, payload.umbral_tipo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
