from io import BytesIO

import numpy as np
import pandas as pd

from app.core.model import ModelStore


def _predict_with_fallback(model, matrix: np.ndarray) -> np.ndarray:
    try:
        return model.predict(matrix)
    except Exception as exc:  # noqa: BLE001 - best-effort fallback for known sklearn mismatch
        message = str(exc)
        if "response_method" not in message:
            raise

        for attr in ("estimator_", "estimator", "base_estimator_", "base_estimator"):
            candidate = getattr(model, attr, None)
            if candidate is not None and hasattr(candidate, "predict"):
                return candidate.predict(matrix)
        raise


def _predict_proba_with_fallback(model, matrix: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(matrix)
        except Exception as exc:  # noqa: BLE001 - surface errors only when unexpected
            message = str(exc)
            if "response_method" not in message:
                raise

    for attr in ("estimator_", "estimator", "base_estimator_", "base_estimator"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "predict_proba"):
            try:
                return candidate.predict_proba(matrix)
            except Exception as exc:  # noqa: BLE001 - keep behavior consistent
                message = str(exc)
                if "response_method" not in message:
                    raise
    return None


def _apply_global_threshold(probabilities: np.ndarray, threshold: float) -> list[float]:
    if probabilities.ndim == 1:
        positive = probabilities
    elif probabilities.shape[1] >= 2:
        positive = probabilities[:, 1]
    else:
        positive = probabilities[:, 0]
    return [float(value >= threshold) for value in positive]


def predict_records(records: list[list[float]]) -> tuple[list[float], list[list[float]] | None]:
    model = ModelStore.get_model()
    matrix = np.asarray(records, dtype=float)
    probabilities = _predict_proba_with_fallback(model, matrix)
    if probabilities is not None:
        threshold = ModelStore.get_bundle().thresholds.get("umbral", 0.5)
        predictions = _apply_global_threshold(probabilities, threshold)
    else:
        predictions = _predict_with_fallback(model, matrix)
    prob_list = probabilities.tolist() if probabilities is not None else None
    return [float(value) for value in predictions], prob_list


def predict_feature_records(
    records: list[dict],
    columns: list[str] | None = None,
) -> tuple[list[float], list[list[float]] | None]:
    if not records:
        raise ValueError("Empty records payload")

    df = pd.DataFrame.from_records(records)
    if columns:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Missing columns: {missing_text}")
        df = df[columns]

    matrix = df.to_numpy(dtype=float)
    model = ModelStore.get_model()
    probabilities = _predict_proba_with_fallback(model, matrix)
    if probabilities is not None:
        threshold = ModelStore.get_bundle().thresholds.get("umbral", 0.5)
        predictions = _apply_global_threshold(probabilities, threshold)
    else:
        predictions = _predict_with_fallback(model, matrix)
    prob_list = probabilities.tolist() if probabilities is not None else None
    return [float(value) for value in predictions], prob_list


def predict_parquet_bytes(
    data: bytes,
    columns: list[str] | None,
) -> tuple[list[float], list[list[float]] | None]:
    if not data:
        raise ValueError("Empty parquet payload")

    df = pd.read_parquet(BytesIO(data))
    if columns:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Missing columns: {missing_text}")
        df = df[columns]
    else:
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=1, how="any")

    if df.empty:
        raise ValueError("No numeric columns available for prediction")

    matrix = df.to_numpy(dtype=float)
    model = ModelStore.get_model()
    probabilities = _predict_proba_with_fallback(model, matrix)
    if probabilities is not None:
        threshold = ModelStore.get_bundle().thresholds.get("umbral", 0.5)
        predictions = _apply_global_threshold(probabilities, threshold)
    else:
        predictions = _predict_with_fallback(model, matrix)
    prob_list = probabilities.tolist() if probabilities is not None else None
    return [float(value) for value in predictions], prob_list