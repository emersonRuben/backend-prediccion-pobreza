# Uso de modelo XGBoost calibrado (PKL)

Este documento explica como cargar y usar el archivo `datos/modelo_xgboost_final.pkl`, incluyendo el umbral por defecto, las variantes, y la **calibración por departamento y año** para reducir el sesgo regional.

## 1) Cargar el modelo

```python
import pickle
from pathlib import Path
import pandas as pd
import json

RUTA_OUT = Path.cwd() / "datos"

with open(RUTA_OUT / "modelo_xgboost_final.pkl", "rb") as f:
    pack = pickle.load(f)

modelo = pack["modelo"]
imputer = pack["imputer"]
features = pack["features"]

umbral_def = pack.get("umbral")
umbral_f1 = pack.get("umbral_f1")
umbral_r80 = pack.get("umbral_r80")

print("Umbral por defecto:", umbral_def)
print("Umbral F1:", umbral_f1)
print("Umbral Recall>=80%:", umbral_r80)
```

## 2) Preparar datos para predecir

El modelo espera las mismas columnas que `features`. Se recomienda mantener el mismo procesamiento numérico usado en el notebook.

```python
# df es un DataFrame con las columnas de entrada
# Asegurar que las columnas existen y son numericas
X = df[features].apply(pd.to_numeric, errors="coerce")
X_i = imputer.transform(X)
```

## 3) Obtener probabilidades

```python
probs = modelo.predict_proba(X_i)[:, 1]
```

## 4) Convertir a clase con umbral

### 4.1) Umbral Global (Simple)
Elige el umbral global según el objetivo:
- `umbral_def`: tasa predicha aproximada a tasa real (política)
- `umbral_f1`: maximiza F1
- `umbral_r80`: asegura recall >= 0.80

```python
# ejemplo con umbral global por defecto
umbral = umbral_def
preds_globales = (probs >= umbral).astype(int)
```

### 4.2) Umbrales Calibrados por Departamento (RECOMENDADO)
Para asegurar que el error predictivo (sesgo regional) se mantenga bajo control (máx 5%) en cada departamento, debes utilizar los archivos JSON de umbrales generados por año (por ejemplo, `umbrales_2024.json`).

```python
# Cargar umbrales del año correspondiente a la petición
anio_peticion = 2024 
with open(RUTA_OUT / f"umbrales_{anio_peticion}.json", "r") as f:
    umbrales_data = json.load(f)

umbrales_dept = umbrales_data["umbrales_por_departamento"]
umbral_global_calibrado = umbrales_data["umbral_global"]

# Aplicar el umbral correspondiente a cada fila (hogar) basándose en su ID de departamento.
# Nota: Los JSON guardan las claves numéricas como strings (ej. "15" para Lima).
def aplicar_umbral_departamento(row):
    if pd.notna(row.get("DEPARTAMENTO")):
        dep_str = str(int(row["DEPARTAMENTO"]))
        umbral_aplicar = umbrales_dept.get(dep_str, umbral_global_calibrado)
    else:
        umbral_aplicar = umbral_global_calibrado
        
    return int(row["prob_pobre"] >= umbral_aplicar)

# Asumiendo que has guardado las probs en tu DataFrame original (ver paso 5)
```

## 5) Ejemplo completo en Endpoint (Backend)

```python
import pickle
import json
from pathlib import Path
import pandas as pd

RUTA_OUT = Path.cwd() / "datos"

# 1. Cargar pack del modelo (se suele hacer al iniciar el servidor)
with open(RUTA_OUT / "modelo_xgboost_final.pkl", "rb") as f:
    pack = pickle.load(f)

modelo = pack["modelo"]
imputer = pack["imputer"]
features = pack["features"]

# 2. Cargar umbrales regionales para el año de análisis (se puede cachear en memoria)
ANIO_ACTUAL = 2024
with open(RUTA_OUT / f"umbrales_{ANIO_ACTUAL}.json", "r") as f:
    umbrales_data = json.load(f)

umbrales_dept = umbrales_data["umbrales_por_departamento"]
umbral_global = umbrales_data["umbral_global"]

# Supongamos que recibimos 'df' de una petición HTTP
# preparar datos
X = df[features].apply(pd.to_numeric, errors="coerce")
X_i = imputer.transform(X)

# probabilidades
probs = modelo.predict_proba(X_i)[:, 1]

# guardar en el DataFrame de salida
out = df.copy()
out["prob_pobre"] = probs

# 3. Aplicar clasificación con calibración departamental
out["pred_pobre"] = out.apply(
    lambda r: int(r["prob_pobre"] >= umbrales_dept.get(str(int(r["DEPARTAMENTO"])), umbral_global))
    if pd.notna(r.get("DEPARTAMENTO")) else int(r["prob_pobre"] >= umbral_global),
    axis=1
)

# Retornar resultados (ej. JSON)
# return out[["id_hogar", "prob_pobre", "pred_pobre"]].to_dict(orient="records")
```

## 6) Notas

- El modelo está calibrado (CalibratedClassifierCV). No necesitas recalibrar las probabilidades al consumirlo.
- Para reducir las brechas y sesgos por región (manteniendo un margen de error menor al 5%), es crítico utilizar la lógica explicada en la sección **4.2**.
- Si faltan columnas en `features`, la predicción fallará. Asegura el mismo esquema de variables.
- Si una petición no contiene el campo `DEPARTAMENTO` o es un departamento desconocido, el código usará el `umbral_global` como método seguro de repliegue (fallback).
