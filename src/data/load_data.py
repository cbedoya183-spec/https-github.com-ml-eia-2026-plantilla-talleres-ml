"""
load_data.py — Carga la muestra ya armada (data/processed/).

Primero intenta leer el Parquet (más rápido); si no existe, usa el CSV.
Si no existe ninguno, explica qué comando correr para generarlo.
"""

import pandas as pd

from src.config import SAMPLE_CSV_PATH, SAMPLE_PARQUET_PATH


def cargar_muestra() -> pd.DataFrame:
    """Devuelve la tabla principal con una fila por vuelo operado."""
    if SAMPLE_PARQUET_PATH.exists():                  # opción 1: Parquet
        return pd.read_parquet(SAMPLE_PARQUET_PATH)
    if SAMPLE_CSV_PATH.exists():                      # opción 2: CSV
        return pd.read_csv(SAMPLE_CSV_PATH, low_memory=False)
    raise FileNotFoundError(                          # ninguna: hay que crear la muestra
        "No existe la muestra. Corre primero:  python src/data/download_data.py"
    )
