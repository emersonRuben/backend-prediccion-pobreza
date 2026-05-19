from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool
    panel_cargado: bool
    n_registros_panel: int


class ModeloInfoResponse(BaseModel):
    anio_test: int | None
    umbral: float
    umbral_f1: float
    umbral_r80: float
    n_features: int
    metricas_test: dict[str, Any] | None


class AniosResponse(BaseModel):
    anios: list[int]


class ResumenAnualResponse(BaseModel):
    anio: int
    tasa_pobreza_total: float
    tasa_pobreza_extrema: float
    n_hogares_estimados: float
    n_pobres_estimados: float
    variacion_vs_anio_anterior: float | None


class RegionalDato(BaseModel):
    departamento_id: int
    departamento: str
    tasa_pobreza: float


class RegionalResponse(BaseModel):
    anio: int
    datos: list[RegionalDato]


class TendenciaNacionalDato(BaseModel):
    anio: int
    tasa_total: float
    tasa_extrema: float


class TendenciaNacionalResponse(BaseModel):
    datos: list[TendenciaNacionalDato]


class TendenciaAreaDato(BaseModel):
    anio: int
    tasa_urbana: float
    tasa_rural: float
    tasa_total: float


class TendenciaAreaResponse(BaseModel):
    datos: list[TendenciaAreaDato]


class TendenciaPoblacionDato(BaseModel):
    anio: int
    pobre_extremo: float
    pobre_no_extremo: float
    no_pobre: float


class TendenciaPoblacionResponse(BaseModel):
    datos: list[TendenciaPoblacionDato]


class TendenciaRegionalDato(BaseModel):
    anio: int
    tasa_pobreza: float


class TendenciaRegionalResponse(BaseModel):
    departamento_id: int
    departamento: str
    datos: list[TendenciaRegionalDato]


class PredictObservacionesRequest(BaseModel):
    observaciones: list[dict[str, Any]] = Field(..., min_length=1)
    umbral_tipo: Literal["umbral", "umbral_f1", "umbral_r80"] = "umbral_f1"


class PredictObservacionesResultado(BaseModel):
    probabilidad_pobreza: float
    clasificacion: Literal["pobre", "no_pobre"]
    umbral_usado: float


class PredictObservacionesResponse(BaseModel):
    resultados: list[PredictObservacionesResultado]
