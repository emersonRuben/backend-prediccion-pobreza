import json
import numpy as np
import pandas as pd
from pathlib import Path

def get_dynamic_threshold_array(df: pd.DataFrame, default_val: float = 0.5, year_col: str = "ANIO", dept_col: str = "DEPARTAMENTO") -> np.ndarray:
    arr = np.full(len(df), default_val, dtype=float)
    if year_col not in df.columns:
        return arr

    for year in df[year_col].dropna().unique():
        try:
            year_int = int(float(year))
        except ValueError:
            continue
            
        threshold_file = Path("thresholds") / f"umbrales_{year_int}.json"
        if not threshold_file.exists():
            continue
            
        with open(threshold_file, "r") as f:
            data = json.load(f)
            
        mask = (df[year_col] == year)
        
        if dept_col in df.columns:
            dept_map = data.get("umbrales_por_departamento", {})
            dept_series = df.loc[mask, dept_col].astype(str).str.replace(r'\.0$', '', regex=True)
            mapped = dept_series.map(dept_map)
            mapped = mapped.fillna(data.get("umbral_global", default_val))
            arr[mask] = mapped.to_numpy(dtype=float)
        else:
            arr[mask] = float(data.get("umbral_global", default_val))
            
    return arr
