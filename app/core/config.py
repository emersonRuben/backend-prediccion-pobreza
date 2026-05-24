from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "enaho-ml-api"
    app_env: str = "dev"
    log_level: str = "INFO"
    model_path: str = "models/modelo_xgboost_final.pkl"
    panel_path: str = "datasets/panel_procesado.parquet"
    model_required: bool = True
    panel_required: bool = True
    cors_origins: list[str] = ["http://localhost:3000"]
    threshold_global: float = 0.5355
    test_year: int | None = 2024
    test_households: int | None = 33691
    test_real_rate_pct: float | None = 20.22
    test_pred_rate_pct: float | None = 20.65
    year_column: str = "ANIO"
    area_column: str = "AREA"
    department_id_column: str = "DEPARTAMENTO"
    department_name_column: str | None = None
    expansion_factor_column: str | None = "FACTOR07"
    feature_columns: list[str] = [
        "AREA",
        "AREA_DOMINIO_freq",
        "DEPARTAMENTO",
        "DOMINIO",
        "ESTRATO",
        "MIEPERHO",
        "P101",
        "P102",
        "P103",
        "P104",
        "P106",
        "P110A1",
        "P1121",
        "P1124",
        "P112A",
        "P1131",
        "P1132",
        "P1141",
        "P1143",
        "P1144",
        "TOTMIEHO",
        "asist_menores",
        "brecha_educ",
        "capital_humano_pct",
        "dummy_covid",
        "edad_jefe",
        "educ_max_pct",
        "educ_min",
        "educ_prom_pct",
        "hacinamiento_pct",
        "indice_privacion_pct",
        "jefe_mujer",
        "n_cronicas",
        "n_informales",
        "n_mayores",
        "n_menores",
        "n_miembros",
        "n_ocupados_pct",
        "n_programas",
        "ratio_dep_pct",
        "sexo_jefe",
        "tasa_analfab",
        "tasa_informalidad",
        "tiene_juntos",
        "tiene_pension65",
        "tiene_qaliwarma",
        "tiene_seguro",
        "ANIO",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("feature_columns", mode="before")
    @classmethod
    def parse_feature_columns(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


settings = Settings()
