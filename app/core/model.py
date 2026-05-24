from dataclasses import dataclass
from typing import Any

from joblib import load


@dataclass(frozen=True)
class ModelBundle:
    estimator: Any
    imputer: Any | None
    features: list[str]
    thresholds: dict[str, float]
    umbrales_dinamicos: dict[int, Any] | None
    metrics: dict[str, Any] | None
    test_year: int | None


class ModelStore:
    _bundle: ModelBundle | None = None

    @classmethod
    def set_bundle(cls, bundle: ModelBundle) -> None:
        cls._bundle = bundle
        
    @classmethod
    def get_bundle(cls) -> ModelBundle:
        if cls._bundle is None:
            raise RuntimeError("Model not loaded")
        return cls._bundle

    @classmethod
    def get_model(cls) -> Any:
        return cls.get_bundle().estimator

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._bundle is not None


def _extract_estimator(payload: Any) -> Any:
    if hasattr(payload, "predict"):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if hasattr(value, "predict"):
                return value
    raise TypeError("Loaded object does not provide a predict method")


def load_model(path: str) -> ModelBundle:
    payload = load(path)
    if isinstance(payload, dict):
        estimator = payload.get("modelo") or _extract_estimator(payload)
        imputer = payload.get("imputer")
        features = list(payload.get("features") or [])
        thresholds = {}
        for key in ("umbral", "umbral_f1", "umbral_r80"):
            value = payload.get(key)
            if value is not None:
                thresholds[key] = float(value)
        metrics = payload.get("metricas_test")
        test_year = payload.get("anio_test")
        umbrales_dinamicos = payload.get("umbrales_dinamicos")
        return ModelBundle(
            estimator=estimator,
            imputer=imputer,
            features=features,
            thresholds=thresholds,
            umbrales_dinamicos=umbrales_dinamicos,
            metrics=metrics,
            test_year=int(test_year) if test_year is not None else None,
        )

    estimator = _extract_estimator(payload)
    return ModelBundle(
        estimator=estimator,
        imputer=None,
        features=[],
        thresholds={},
        umbrales_dinamicos=None,
        metrics=None,
        test_year=None,
    )
