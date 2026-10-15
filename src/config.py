"""
config.py — Todas las constantes del proyecto en un solo lugar.

Si hay que cambiar una ruta, la semilla, el periodo de estudio o la lista de
columnas, se cambia AQUÍ y el resto del código lo toma automáticamente.
"""

from pathlib import Path

# =============================================================================
# 1. RUTAS DEL PROYECTO
# =============================================================================
# Path(__file__) es la ruta de este archivo (src/config.py).
# .resolve().parent sube a "src" y el segundo .parent sube a la raíz del repo.
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data"                  # carpeta de datos
RAW_DATA_DIR = DATA_DIR / "raw"               # 12 CSV mensuales tal como vienen de Kaggle
PROCESSED_DATA_DIR = DATA_DIR / "processed"   # muestra ya armada por nosotros
MODELS_DIR = BASE_DIR / "models"              # modelo guardado y tablas de resultados
REPORTS_DIR = BASE_DIR / "reports"            # auditorías y figuras
FIGURES_DIR = REPORTS_DIR / "figuras"         # gráficas que se usan en la exposición

# Archivo con la muestra reproducible (1 % de los vuelos operados de cada mes)
SAMPLE_PARQUET_PATH = PROCESSED_DATA_DIR / "vuelos_muestra.parquet"
SAMPLE_CSV_PATH = PROCESSED_DATA_DIR / "vuelos_muestra.csv"

# =============================================================================
# 2. FUENTE DE DATOS (Kaggle, que reproduce los datos oficiales del BTS / DOT)
# =============================================================================
KAGGLE_DATASET = "umerhafeezsiddiqui/2021-2026-us-flights"
KAGGLE_URL = "https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights"
BTS_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ"  # tabla oficial "Reporting Carrier On-Time Performance"

# Periodo de estudio: 12 meses completos (junio 2025 a mayo 2026) = un ciclo estacional.
MESES_ESTUDIO = [
    "Jun-2025.csv", "Jul-2025.csv", "Aug-2025.csv", "Sep-2025.csv",
    "Oct-2025.csv", "Nov-2025.csv", "Dec-2025.csv", "Jan-2026.csv",
    "Feb-2026.csv", "Mar-2026.csv", "Apr-2026.csv", "May-2026.csv",
]

# =============================================================================
# 3. REPRODUCIBILIDAD Y VALIDACIÓN
# =============================================================================
SEMILLA = 42             # misma semilla en todo: los números coinciden en cada corrida
FRACCION_MUESTRA = 0.01  # 1 % de los vuelos operados de cada mes (~76 mil filas)
FRACCION_PRUEBA = 0.25   # el último 25 % (en el tiempo) se reserva para la prueba final
PLIEGUES_CV = 3          # pliegues de TimeSeriesSplit para comparar modelos
PLIEGUES_AJUSTE = 2      # pliegues para la búsqueda de hiperparámetros
MARGEN_SIMPLICIDAD = 0.01  # si otro modelo queda a menos de 0,01 de AUC, gana el más simple
GANANCIA_MINIMA = 0.005    # ganancia mínima de AUC para aceptar hiperparámetros nuevos

# =============================================================================
# 4. VARIABLE OBJETIVO
# =============================================================================
# ARR_DEL15 = 1 si el vuelo llegó 15 minutos o más tarde (estándar FAA / DOT).
TARGET = "ARR_DEL15"
UMBRAL_MINUTOS = 15

# =============================================================================
# 5. COLUMNAS: cuáles se usan, cuáles son fuga y cuáles son administrativas
# =============================================================================
# Columnas que se leen del CSV crudo (el archivo trae más, pero no las necesitamos).
COLUMNAS_A_LEER = [
    # --- identificación y fecha ---
    "FL_DATE", "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM",
    "ORIGIN", "DEST", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID",
    # --- itinerario programado (se conoce antes del vuelo) ---
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE",
    # --- resultado del vuelo (se leen solo para auditar fugas y crear el target) ---
    "DEP_TIME", "DEP_DELAY", "DEP_DEL15", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON",
    "TAXI_IN", "ARR_TIME", "ARR_DELAY", "ARR_DEL15", "AIR_TIME", "ACTUAL_ELAPSED_TIME",
    "CANCELLED", "DIVERTED",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]

# FUGA DE INFORMACIÓN: columnas que solo se conocen DESPUÉS de que el avión sale
# o aterriza. Si entran al modelo, este "hace trampa" y en producción no sirve.
COLUMNAS_FUGA = [
    "DEP_TIME", "DEP_DELAY", "DEP_DEL15", "TAXI_OUT",          # al salir de puerta
    "WHEELS_OFF", "WHEELS_ON", "AIR_TIME", "ACTUAL_ELAPSED_TIME",  # en el aire
    "TAXI_IN", "ARR_TIME", "ARR_DELAY",                         # al llegar
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",              # causas calculadas
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",                    # después del vuelo
]

# Identificadores que no generalizan (o repiten información de otra columna).
COLUMNAS_ADMIN = ["TAIL_NUM", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID"]

# Los 10 aeropuertos con más tráfico de EE.UU. (FAA). Se usan en la regla manual
# y en las columnas ORIGIN_IS_HUB / DEST_IS_HUB.
HUBS = {"ATL", "ORD", "DFW", "DEN", "CLT", "LAX", "JFK", "MIA", "MCO", "EWR"}

# Columnas que entran al modelo, separadas por tipo (cada tipo recibe un tratamiento).
COLUMNAS_NUMERICAS = [
    "CRS_DEP_HOUR", "CRS_ARR_HOUR", "CRS_ELAPSED_TIME", "DISTANCE",
    "DAY_OF_WEEK", "DAY_OF_MONTH", "MONTH", "IS_WEEKEND",
    "ORIGIN_IS_HUB", "DEST_IS_HUB", "IS_HUB_TO_HUB",
    "CRS_DEP_SIN", "CRS_DEP_COS", "MONTH_SIN", "MONTH_COS",
]
COLUMNAS_CATEGORICAS = [
    "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
    "DEP_TIME_BLOCK", "DISTANCE_CAT", "SEASON",
]
# La aerolínea va también en la lista porque el imputador por grupo la necesita.
COLUMNAS_MODELO = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS

# Aeropuertos más frecuentes que conservan su nombre; el resto pasa a "OTRO".
TOP_AEROPUERTOS = 25
