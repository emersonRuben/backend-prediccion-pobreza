from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.model import ModelStore
from app.core.panel import PanelStore


def _get_feature_columns() -> list[str]:
    bundle = ModelStore.get_bundle()
    return bundle.features or settings.feature_columns

 
def _get_threshold(name: str) -> float:  #Se modifica el umbral de predicción
    bundle = ModelStore.get_bundle()
    return float(bundle.thresholds.get(name, 0.5355))


def _get_department_name_column(panel: pd.DataFrame) -> str | None:
    if settings.department_name_column and settings.department_name_column in panel.columns:
        return settings.department_name_column
    for candidate in ("DEPARTAMENTO_NOMBRE", "DEPARTAMENTO_NAME"):
        if candidate in panel.columns:
            return candidate
    return None


DEPARTMENT_NAMES: dict[int, str] = {
    1: "Amazonas",
    2: "Ancash",
    3: "Apurimac",
    4: "Arequipa",
    5: "Ayacucho",
    6: "Cajamarca",
    7: "Callao",
    8: "Cusco",
    9: "Huancavelica",
    10: "Huanuco",
    11: "Ica",
    12: "Junin",
    13: "La Libertad",
    14: "Lambayeque",
    15: "Lima Provincias",
    16: "Loreto",
    17: "Madre de Dios",
    18: "Moquegua",
    19: "Pasco",
    20: "Piura",
    21: "Puno",
    22: "San Martin",
    23: "Tacna",
    24: "Tumbes",
    25: "Ucayali",
}


def _get_expansion_column(panel: pd.DataFrame) -> str | None:
    if settings.expansion_factor_column and settings.expansion_factor_column in panel.columns:
        return settings.expansion_factor_column
    return None


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(values)) if values.size else 0.0
    total = float(np.sum(weights))
    if total == 0:
        return 0.0
    return float(np.sum(values * weights) / total)


def _weighted_sum(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.sum(values))
    return float(np.sum(values * weights))


def _prepare_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [column for column in features if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing columns: {missing_text}")

    data = df[features].apply(pd.to_numeric, errors="coerce")
    bundle = ModelStore.get_bundle()
    if bundle.imputer is not None:
        return bundle.imputer.transform(data)
    return data.to_numpy(dtype=float)


def _predict_probabilities(df: pd.DataFrame) -> np.ndarray:
    features = _get_feature_columns()
    matrix = _prepare_matrix(df, features)
    estimator = ModelStore.get_model()
    candidates = [
        estimator,
        getattr(estimator, "estimator_", None),
        getattr(estimator, "estimator", None),
        getattr(estimator, "base_estimator_", None),
        getattr(estimator, "base_estimator", None),
    ]

    for candidate in [item for item in candidates if item is not None]:
        if hasattr(candidate, "predict_proba"):
            try:
                probabilities = candidate.predict_proba(matrix)
                return np.asarray(probabilities)[:, 1]
            except Exception:  # noqa: BLE001 - fallback to predict for incompatible estimators
                pass

        if hasattr(candidate, "predict"):
            try:
                predictions = candidate.predict(matrix)
                return np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)
            except Exception:  # noqa: BLE001 - try other candidates
                pass

    raise RuntimeError("No compatible estimator available for prediction")


def _ensure_panel() -> pd.DataFrame:
    if not PanelStore.is_loaded():
        raise RuntimeError("Panel not loaded")
    return PanelStore.get_panel()


def get_years() -> list[int]:
    panel = _ensure_panel()
    years = panel[settings.year_column].dropna().unique().tolist()
    return sorted(int(value) for value in years)


def summarize_year(anio: int) -> dict[str, Any]:
    panel = _ensure_panel()
    year_column = settings.year_column
    filtered = panel[panel[year_column] == anio]
    if filtered.empty:
        raise ValueError("No data for requested year")

    probabilities = _predict_probabilities(filtered)
    from app.core.thresholds import get_dynamic_threshold_array
    threshold_total_arr = get_dynamic_threshold_array(filtered, default_val=_get_threshold("umbral"))
    threshold_extreme = _get_threshold("umbral_r80")

    expansion_col = _get_expansion_column(filtered)
    weights = filtered[expansion_col].to_numpy(dtype=float) if expansion_col else None

    total_rate = _weighted_mean((probabilities >= threshold_total_arr).astype(float), weights)
    extreme_rate = _weighted_mean((probabilities >= threshold_extreme).astype(float), weights)

    total_households = _weighted_sum(np.ones_like(probabilities), weights)
    poor_households = _weighted_sum((probabilities >= threshold_total_arr).astype(float), weights)

    previous_year = anio - 1
    variation = None
    if previous_year in get_years():
        previous = summarize_year(previous_year)
        variation = float(total_rate - previous["tasa_pobreza_total"])

    return {
        "anio": anio,
        "tasa_pobreza_total": total_rate,
        "tasa_pobreza_extrema": extreme_rate,
        "n_hogares_estimados": total_households,
        "n_pobres_estimados": poor_households,
        "variacion_vs_anio_anterior": variation,
    }


def regional_rates(anio: int, top_n: int | None = None) -> dict[str, Any]:
    panel = _ensure_panel()
    filtered = panel[panel[settings.year_column] == anio]
    if filtered.empty:
        raise ValueError("No data for requested year")

    department_col = settings.department_id_column
    name_col = _get_department_name_column(filtered)
    from app.core.thresholds import get_dynamic_threshold_array

    results = []
    for department_id, group in filtered.groupby(department_col):
        probabilities = _predict_probabilities(group)
        threshold_total_arr = get_dynamic_threshold_array(group, default_val=_get_threshold("umbral"))
        expansion_col = _get_expansion_column(group)
        weights = group[expansion_col].to_numpy(dtype=float) if expansion_col else None
        rate = _weighted_mean((probabilities >= threshold_total_arr).astype(float), weights)
        department_name = None
        if name_col:
            department_name = group[name_col].iloc[0]
        else:
            department_name = DEPARTMENT_NAMES.get(int(department_id))
        results.append(
            {
                "departamento_id": int(department_id),
                "departamento": str(department_name) if department_name is not None else str(department_id),
                "tasa_pobreza": rate,
            }
        )

    results.sort(key=lambda item: item["tasa_pobreza"], reverse=True)
    if top_n:
        results = results[:top_n]

    return {"anio": anio, "datos": results}


def national_trend() -> dict[str, Any]:
    data = []
    for year in get_years():
        summary = summarize_year(year)
        data.append(
            {
                "anio": year,
                "tasa_total": summary["tasa_pobreza_total"],
                "tasa_extrema": summary["tasa_pobreza_extrema"],
            }
        )
    return {"datos": data}


def area_trend() -> dict[str, Any]:
    panel = _ensure_panel()
    from app.core.thresholds import get_dynamic_threshold_array
    area_col = settings.area_column

    data = []
    for year in get_years():
        subset = panel[panel[settings.year_column] == year]
        if subset.empty:
            continue

        total_prob = _predict_probabilities(subset)
        threshold_total_arr = get_dynamic_threshold_array(subset, default_val=_get_threshold("umbral"))
        expansion_col = _get_expansion_column(subset)
        weights_total = subset[expansion_col].to_numpy(dtype=float) if expansion_col else None
        total_rate = _weighted_mean((total_prob >= threshold_total_arr).astype(float), weights_total)

        urban_rate = 0.0
        rural_rate = 0.0
        for area_value, label in ((1, "urban"), (2, "rural")):
            area_subset = subset[subset[area_col] == area_value]
            if area_subset.empty:
                continue
            probs = _predict_probabilities(area_subset)
            area_threshold_arr = get_dynamic_threshold_array(area_subset, default_val=_get_threshold("umbral"))
            expansion_col_a = _get_expansion_column(area_subset)
            weights = area_subset[expansion_col_a].to_numpy(dtype=float) if expansion_col_a else None
            rate = _weighted_mean((probs >= area_threshold_arr).astype(float), weights)
            if label == "urban":
                urban_rate = rate
            else:
                rural_rate = rate

        data.append(
            {
                "anio": year,
                "tasa_urbana": urban_rate,
                "tasa_rural": rural_rate,
                "tasa_total": total_rate,
            }
        )

    return {"datos": data}


def population_trend() -> dict[str, Any]:
    panel = _ensure_panel()
    from app.core.thresholds import get_dynamic_threshold_array
    threshold_extreme = _get_threshold("umbral_r80")

    data = []
    for year in get_years():
        subset = panel[panel[settings.year_column] == year]
        if subset.empty:
            continue

        probabilities = _predict_probabilities(subset)
        threshold_total_arr = get_dynamic_threshold_array(subset, default_val=_get_threshold("umbral"))
        expansion_col = _get_expansion_column(subset)
        weights = subset[expansion_col].to_numpy(dtype=float) if expansion_col else None

        poor_extreme = _weighted_sum((probabilities >= threshold_extreme).astype(float), weights)
        poor_total = _weighted_sum((probabilities >= threshold_total_arr).astype(float), weights)
        poor_no_extreme = max(poor_total - poor_extreme, 0.0)
        no_poor = _weighted_sum((probabilities < threshold_total_arr).astype(float), weights)

        data.append(
            {
                "anio": year,
                "pobre_extremo": poor_extreme,
                "pobre_no_extremo": poor_no_extreme,
                "no_pobre": no_poor,
            }
        )

    return {"datos": data}


def regional_trend(department_id: int) -> dict[str, Any]:
    panel = _ensure_panel()
    department_col = settings.department_id_column
    from app.core.thresholds import get_dynamic_threshold_array
    
    data = []
    department_name = None
    for year in get_years():
        subset = panel[(panel[settings.year_column] == year) & (panel[department_col] == department_id)]
        if subset.empty:
            continue
        probabilities = _predict_probabilities(subset)
        threshold_total_arr = get_dynamic_threshold_array(subset, default_val=_get_threshold("umbral"))
        expansion_col = _get_expansion_column(subset)
        weights = subset[expansion_col].to_numpy(dtype=float) if expansion_col else None
        rate = _weighted_mean((probabilities >= threshold_total_arr).astype(float), weights)

        name_col = _get_department_name_column(subset)
        if name_col and department_name is None:
            department_name = subset[name_col].iloc[0]
        if department_name is None:
            department_name = DEPARTMENT_NAMES.get(int(department_id))

        data.append({"anio": year, "tasa_pobreza": rate})

    return {
        "departamento_id": int(department_id),
        "departamento": str(department_name) if department_name is not None else str(department_id),
        "datos": data,
    }


def predict_observations(records: list[dict[str, Any]], threshold_name: str) -> dict[str, Any]:
    if not records:
        raise ValueError("Empty records payload")

    df = pd.DataFrame.from_records(records)
    probabilities = _predict_probabilities(df)
    threshold = _get_threshold(threshold_name)

    resultados = []
    for probability in probabilities:
        classification = "pobre" if probability >= threshold else "no_pobre"
        resultados.append(
            {
                "probabilidad_pobreza": float(probability),
                "clasificacion": classification,
                "umbral_usado": float(threshold),
            }
        )

    return {"resultados": resultados}
