"""
build_features.py — PASOS 3 y 4: limpieza de filas y creación de columnas.

Regla de oro: aquí SOLO van transformaciones que no "aprenden" nada de los datos
(no calculan medias, frecuencias ni nada que dependa de otras filas). Cada fila
se transforma mirando únicamente sus propios valores. Por eso se pueden aplicar
antes de separar entrenamiento y prueba sin causar fuga de información.

Todo lo que SÍ aprende de los datos (imputar con medianas, escalar, one-hot)
vive dentro del Pipeline de scikit-learn (src/models/pipeline.py).
"""

import numpy as np
import pandas as pd

from src.config import COLUMNAS_ADMIN, COLUMNAS_FUGA, HUBS, TARGET
from src.utils.logger import get_logger

logger = get_logger("build_features")


# -----------------------------------------------------------------------------
# Utilidad: convertir la fecha de texto a fecha real
# -----------------------------------------------------------------------------
def convertir_fecha(columna: pd.Series) -> pd.Series:
    """Convierte FL_DATE (texto) a fecha real.

    En los CSV de Kaggle la fecha llega en DOS formatos mezclados según el mes:
        "6/30/2025 12:00:00 AM"   -> mes/día/año(4) hora AM/PM
        "06-01-25 00:00"          -> mes-día-año(2) hora 24 h
    Se prueba cada formato exacto en orden y cada uno llena solo los vacíos que
    dejó el anterior. Lo que quede sin leer se intenta con format="mixed".
    """
    if pd.api.types.is_datetime64_any_dtype(columna):     # ya es fecha: no hay nada que hacer
        return columna
    texto = columna.astype(str).str.strip()
    fecha = pd.Series(pd.NaT, index=columna.index, dtype="datetime64[ns]")
    for formato in ["%m/%d/%Y %I:%M:%S %p", "%m-%d-%y %H:%M", "%Y-%m-%d", "%m/%d/%Y"]:
        faltan = fecha.isna()                              # filas que todavía no se pudieron leer
        if not faltan.any():
            break
        fecha[faltan] = pd.to_datetime(texto[faltan], format=formato, errors="coerce")
    faltan = fecha.isna()
    if faltan.any():                                       # último recurso: pandas deduce fila por fila
        fecha[faltan] = pd.to_datetime(texto[faltan], format="mixed", errors="coerce")
    return fecha


# -----------------------------------------------------------------------------
# PASO 3 · LIMPIEZA DE FILAS
# -----------------------------------------------------------------------------
def limpiar_filas(df: pd.DataFrame) -> pd.DataFrame:
    """Deja solo vuelos operados, con etiqueta, sin duplicados y ordenados por fecha."""
    df = df.copy()                                         # nunca modificar la tabla original
    n_inicial = len(df)

    # 1) Cancelados y desviados no llegan en su itinerario: no tienen etiqueta válida.
    #    (La muestra ya viene sin ellos; se repite el filtro por seguridad.)
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]

    # 2) Sin ARR_DEL15 no sabemos si el vuelo se retrasó: no se puede imputar la etiqueta.
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)                    # 0.0/1.0 -> 0/1

    # 3) Duplicados: el mismo vuelo (día, aerolínea, número, origen, hora) dos veces.
    clave = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "CRS_DEP_TIME"]
    df = df.drop_duplicates(subset=[c for c in clave if c in df.columns])

    # 4) La fecha llega como texto ("6/1/2025 12:00:00 AM"); se convierte a fecha real
    #    para poder ordenar y separar entrenamiento/prueba en el tiempo.
    df["FL_DATE"] = convertir_fecha(df["FL_DATE"])
    df = df.dropna(subset=["FL_DATE"])

    # 5) Orden cronológico (fecha y hora programada). El corte de prueba depende de esto.
    df = df.sort_values(["FL_DATE", "CRS_DEP_TIME"], kind="mergesort").reset_index(drop=True)

    logger.info(f"Limpieza: {n_inicial:,} -> {len(df):,} filas "
                f"(tasa de retraso {df[TARGET].mean():.2%})")
    return df


# -----------------------------------------------------------------------------
# PASO 4 · COLUMNAS NUEVAS (todas con información disponible antes del vuelo)
# -----------------------------------------------------------------------------
def hhmm_a_hora(valor) -> float:
    """Convierte la hora programada en formato HHMM (p. ej. 1435) a la hora entera (14)."""
    if pd.isna(valor):                                     # vacío -> se deja vacío (lo imputa el Pipeline)
        return np.nan
    hora = int(float(valor)) // 100                        # 1435 // 100 = 14
    if hora == 24:                                         # el BTS escribe medianoche como 2400
        return 0.0
    return float(hora) if 0 <= hora <= 23 else np.nan      # cualquier valor raro queda vacío


def estacion(mes: int) -> str:
    """Estación del año en EE.UU. según el mes (clima de invierno vs. tormentas de verano)."""
    if mes in (12, 1, 2):
        return "Invierno"
    if mes in (3, 4, 5):
        return "Primavera"
    if mes in (6, 7, 8):
        return "Verano"
    return "Otono"


def franja_horaria(hora: float) -> str | float:
    """Bloque del día: los retrasos se acumulan a lo largo del día (efecto cascada)."""
    if pd.isna(hora):
        return np.nan
    if hora < 6:
        return "Madrugada"
    if hora < 12:
        return "Manana"
    if hora < 18:
        return "Tarde"
    return "Noche"


def categoria_distancia(millas: float) -> str | float:
    """Corta (<500 mi), media (500-1500 mi) o larga (>1500 mi)."""
    if pd.isna(millas):
        return np.nan
    if millas < 500:
        return "Corta"
    if millas <= 1500:
        return "Media"
    return "Larga"


def crear_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las columnas de calendario, horario y red de aeropuertos."""
    df = df.copy()

    # --- Calendario -----------------------------------------------------------
    df["MONTH"] = df["FL_DATE"].dt.month                   # 1..12
    df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day              # 1..31
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek         # 0 = lunes ... 6 = domingo
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)  # sábado o domingo
    df["SEASON"] = df["MONTH"].apply(estacion)

    # --- Horario programado ---------------------------------------------------
    df["CRS_DEP_HOUR"] = df["CRS_DEP_TIME"].apply(hhmm_a_hora)  # hora de salida (0-23)
    df["CRS_ARR_HOUR"] = df["CRS_ARR_TIME"].apply(hhmm_a_hora)  # hora de llegada (0-23)
    df["DEP_TIME_BLOCK"] = df["CRS_DEP_HOUR"].apply(franja_horaria)

    # Codificación cíclica: en números, 23 y 0 parecen lejanos, pero son horas vecinas.
    # Con seno y coseno la hora queda sobre un círculo y 23 h queda junto a 0 h
    # (lo mismo con diciembre y enero). Es una fórmula fija: no aprende nada.
    df["CRS_DEP_SIN"] = np.sin(2 * np.pi * df["CRS_DEP_HOUR"] / 24)
    df["CRS_DEP_COS"] = np.cos(2 * np.pi * df["CRS_DEP_HOUR"] / 24)
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    # --- Red de aeropuertos ---------------------------------------------------
    df["ORIGIN_IS_HUB"] = df["ORIGIN"].isin(HUBS).astype(int)  # sale de un aeropuerto congestionado
    df["DEST_IS_HUB"] = df["DEST"].isin(HUBS).astype(int)      # llega a un aeropuerto congestionado
    df["IS_HUB_TO_HUB"] = df["ORIGIN_IS_HUB"] * df["DEST_IS_HUB"]  # 1 solo si ambos son hub
    df["DISTANCE_CAT"] = df["DISTANCE"].apply(categoria_distancia)

    return df


def descartar_fugas(df: pd.DataFrame) -> pd.DataFrame:
    """Quita las columnas que se conocen después del vuelo y los identificadores."""
    a_quitar = [c for c in COLUMNAS_FUGA + COLUMNAS_ADMIN if c in df.columns]
    vacias = [c for c in df.columns if df[c].isna().all()]     # columnas 100 % vacías
    logger.info(f"Se descartan {len(a_quitar)} columnas de fuga/identificadores "
                f"y {len(vacias)} columnas vacías")
    return df.drop(columns=a_quitar + vacias)


def preparar_tabla(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """Atajo: limpieza + columnas nuevas + descarte de fugas, en ese orden."""
    return descartar_fugas(crear_columnas(limpiar_filas(df_crudo)))
