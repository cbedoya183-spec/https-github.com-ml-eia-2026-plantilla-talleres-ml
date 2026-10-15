"""
download_data.py — PASO 2 (Datos): descarga de los 12 CSV y armado de la muestra.

Qué hace, en orden:
1. Si faltan CSV en data/raw/, los descarga de Kaggle con la API oficial.
2. Lee cada mes, cuenta cancelados, desviados, vacíos y duplicados (auditoría).
3. De los vuelos OPERADOS de cada mes toma el 1 % al azar (semilla 42).
   Un mes con más vuelos aporta más filas: la muestra respeta el peso de cada mes.
4. Une los 12 meses en una sola tabla y la guarda en data/processed/.

Uso desde la terminal (en la raíz del repositorio):
    python src/data/download_data.py                  # 1 % por mes (valor oficial)
    python src/data/download_data.py --fraccion 0.02  # otra fracción
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Permite correr este archivo directamente (python src/data/download_data.py)
# aunque el paquete no esté instalado: agrega la raíz del repo a la ruta de búsqueda.
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import (  # noqa: E402  (el import va después de ajustar sys.path)
    COLUMNAS_A_LEER,
    FRACCION_MUESTRA,
    KAGGLE_DATASET,
    MESES_ESTUDIO,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SAMPLE_CSV_PATH,
    SAMPLE_PARQUET_PATH,
    SEMILLA,
    TARGET,
)
from src.features.build_features import convertir_fecha  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("download_data")

# Columnas que identifican un vuelo único: mismo día, aerolínea, número, origen y hora.
CLAVE_VUELO = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "CRS_DEP_TIME"]


# -----------------------------------------------------------------------------
# 1. DESCARGA
# -----------------------------------------------------------------------------
def descargar_de_kaggle(archivos: list[str]) -> None:
    """Descarga de Kaggle los CSV mensuales que todavía no están en data/raw/."""
    faltantes = [a for a in archivos if not (RAW_DATA_DIR / a).exists()]  # solo los que no están
    if not faltantes:                                   # si ya están todos, no hace nada
        logger.info("Los 12 CSV ya están en data/raw/. No se descarga nada.")
        return

    # La API de Kaggle busca el token en ~/.kaggle/kaggle.json o en variables de entorno.
    # Cómo obtenerlo: kaggle.com > Settings > API > "Create New Token".
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:                        # la librería no está instalada
        raise SystemExit("Instala la librería: pip install kaggle") from error

    api = KaggleApi()
    api.authenticate()                                  # falla con un mensaje claro si no hay token
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)     # crea data/raw/ si no existe

    for archivo in faltantes:
        logger.info(f"Descargando {archivo} desde Kaggle ({KAGGLE_DATASET}) ...")
        api.dataset_download_file(KAGGLE_DATASET, archivo, path=str(RAW_DATA_DIR))
        comprimido = RAW_DATA_DIR / f"{archivo}.zip"    # Kaggle a veces entrega el archivo en .zip
        if comprimido.exists():
            import zipfile
            with zipfile.ZipFile(comprimido) as zf:     # descomprimir y borrar el .zip
                zf.extractall(RAW_DATA_DIR)
            os.remove(comprimido)


# -----------------------------------------------------------------------------
# 2. LECTURA Y AUDITORÍA DE UN MES
# -----------------------------------------------------------------------------
def leer_mes(ruta: Path) -> pd.DataFrame:
    """Lee un CSV mensual con solo las columnas que el proyecto necesita."""
    return pd.read_csv(
        ruta,
        usecols=lambda c: c in COLUMNAS_A_LEER,  # ignora columnas que no usamos (ahorra memoria)
        low_memory=False,                       # lee el archivo completo antes de decidir tipos
    )


def auditar_mes(df: pd.DataFrame, archivo: str) -> dict:
    """Cuenta lo que hay que limpiar en un mes crudo. Devuelve una fila de la auditoría."""
    cancelados = int((df["CANCELLED"] == 1).sum())        # vuelos que nunca salieron
    desviados = int((df["DIVERTED"] == 1).sum())          # vuelos que aterrizaron en otro lado
    operados = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0) & df[TARGET].notna()]
    fechas = convertir_fecha(df["FL_DATE"])                 # para ver el rango de fechas del archivo
    clave = [c for c in CLAVE_VUELO if c in df.columns]   # columnas de la clave que existen
    return {
        "archivo": archivo,
        "filas": len(df),
        "cancelados": cancelados,
        "desviados": desviados,
        "nulos_target": int(df[TARGET].isna().sum()),      # sin ARR_DEL15 (no hay etiqueta)
        "operados_con_target": len(operados),
        "tasa_retraso_operados": round(float(operados[TARGET].mean()), 4),
        "nulos_crs_dep_time": int(operados["CRS_DEP_TIME"].isna().sum()),
        "nulos_crs_elapsed": int(operados["CRS_ELAPSED_TIME"].isna().sum()),
        "nulos_distance": int(operados["DISTANCE"].isna().sum()),
        "duplicados_clave_vuelo": int(operados.duplicated(subset=clave).sum()),
        "fecha_min": fechas.min().date().isoformat(),
        "fecha_max": fechas.max().date().isoformat(),
    }


def muestrear_mes(df: pd.DataFrame, fraccion: float) -> pd.DataFrame:
    """Toma al azar una fracción de los vuelos OPERADOS de un mes (semilla fija)."""
    # Un vuelo cancelado o desviado no tiene llegada en su itinerario: no tiene etiqueta.
    operados = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0) & df[TARGET].notna()]
    # sample() elige filas sin reemplazo; random_state fija el sorteo para que sea repetible.
    return operados.sample(frac=fraccion, random_state=SEMILLA)


# -----------------------------------------------------------------------------
# 3. ARMADO DE LA TABLA PRINCIPAL
# -----------------------------------------------------------------------------
def construir_muestra(fraccion: float = FRACCION_MUESTRA, descargar: bool = True) -> pd.DataFrame:
    """Recorre los 12 meses, audita cada uno, toma la muestra y guarda todo."""
    if descargar:
        descargar_de_kaggle(MESES_ESTUDIO)              # paso opcional: solo si faltan archivos

    disponibles = [a for a in MESES_ESTUDIO if (RAW_DATA_DIR / a).exists()]
    if not disponibles:                                 # sin archivos no se puede seguir
        raise SystemExit(
            f"No hay CSV en {RAW_DATA_DIR}. Descárgalos de Kaggle y colócalos ahí "
            "(Jun-2025.csv ... May-2026.csv)."
        )

    auditoria, muestras, cobertura = [], [], []         # listas donde se acumula cada mes
    for archivo in disponibles:
        logger.info(f"Procesando {archivo} ...")
        df_mes = leer_mes(RAW_DATA_DIR / archivo)       # 1) leer el mes completo
        fila = auditar_mes(df_mes, archivo)             # 2) auditar lo crudo
        muestra = muestrear_mes(df_mes, fraccion)       # 3) tomar el 1 % de los operados
        auditoria.append(fila)
        muestras.append(muestra)
        cobertura.append({                              # cuánto aportó el mes a la muestra
            "archivo": archivo,
            "operados": fila["operados_con_target"],
            "muestra": len(muestra),
            "fraccion": round(len(muestra) / fila["operados_con_target"], 4),
            "tasa_retraso_operados": fila["tasa_retraso_operados"],
            "tasa_retraso_muestra": round(float(muestra[TARGET].mean()), 4),
        })
        logger.info(f"  {fila['filas']:,} filas crudas -> {len(muestra):,} en la muestra")
        del df_mes                                      # libera memoria antes del siguiente mes

    tabla = pd.concat(muestras, ignore_index=True)      # une los 12 meses en una sola tabla

    # Guardar resultados: la muestra (datos) y las dos tablas de auditoría (evidencia).
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(SAMPLE_PARQUET_PATH, index=False)  # Parquet: rápido y conserva los tipos
    tabla.to_csv(SAMPLE_CSV_PATH, index=False)          # CSV: para abrirlo en Excel si se quiere
    pd.DataFrame(auditoria).to_csv(REPORTS_DIR / "auditoria_mensual_cruda.csv", index=False)
    pd.DataFrame(cobertura).to_csv(REPORTS_DIR / "muestra_vs_mes.csv", index=False)

    logger.info(f"Muestra final: {len(tabla):,} filas x {tabla.shape[1]} columnas")
    logger.info(f"Guardada en {SAMPLE_PARQUET_PATH}")
    return tabla


# -----------------------------------------------------------------------------
# Punto de entrada cuando se ejecuta desde la terminal
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga y muestra de vuelos de EE.UU.")
    parser.add_argument("--fraccion", type=float, default=FRACCION_MUESTRA,
                        help="fracción de vuelos operados que se toma de cada mes (0.01 = 1 %%)")
    parser.add_argument("--sin-descarga", action="store_true",
                        help="no intenta descargar de Kaggle; usa lo que haya en data/raw/")
    args = parser.parse_args()
    construir_muestra(fraccion=args.fraccion, descargar=not args.sin_descarga)
