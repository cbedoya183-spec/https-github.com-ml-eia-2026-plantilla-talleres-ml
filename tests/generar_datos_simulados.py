"""
generar_datos_simulados.py — Datos INVENTADOS con el mismo formato de los CSV del BTS.

Sirven SOLO para probar que el código corre de principio a fin sin tener que
descargar 1 GB de Kaggle. Los números que salen con estos datos NO son
resultados del proyecto: los resultados oficiales se obtienen con los CSV reales.

Uso:
    python tests/generar_datos_simulados.py --destino carpeta_prueba --filas 20000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MESES = [("Jun", 2025, 6), ("Jul", 2025, 7), ("Aug", 2025, 8), ("Sep", 2025, 9),
         ("Oct", 2025, 10), ("Nov", 2025, 11), ("Dec", 2025, 12), ("Jan", 2026, 1),
         ("Feb", 2026, 2), ("Mar", 2026, 3), ("Apr", 2026, 4), ("May", 2026, 5)]
AEROLINEAS = ["WN", "AA", "DL", "UA", "OO", "YX", "B6", "AS", "NK", "F9", "G4", "HA", "MQ", "OH"]
AEROPUERTOS = ["ATL", "ORD", "DFW", "DEN", "CLT", "LAX", "JFK", "MIA", "MCO", "EWR", "LAS", "PHX",
               "SEA", "SFO", "IAH", "BOS", "MSP", "DTW", "PHL", "LGA", "BWI", "SLC", "SAN", "TPA",
               "DCA", "MDW", "BNA", "AUS", "HOU", "PDX", "STL", "RDU", "MSY", "SMF", "SJC", "OAK",
               "CLE", "PIT", "IND", "CMH", "MCI", "SAT", "RSW", "BUF", "ABQ", "OMA", "BOI", "TUS"]
HUBS = {"ATL", "ORD", "DFW", "DEN", "CLT", "LAX", "JFK", "MIA", "MCO", "EWR"}
EFECTO_MES = {6: 0.45, 7: 0.45, 8: 0.1, 9: -0.4, 10: -0.15, 11: -0.1, 12: 0.3,
              1: 0.0, 2: -0.1, 3: 0.15, 4: -0.1, 5: 0.0}


def generar_mes(abrev: str, anio: int, mes: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    dias = pd.Period(f"{anio}-{mes:02d}").days_in_month
    fecha = pd.to_datetime({"year": anio, "month": mes, "day": rng.integers(1, dias + 1, n)})
    pesos = np.linspace(3, 0.3, len(AEROPUERTOS))
    pesos /= pesos.sum()                                          # aeropuertos grandes más frecuentes
    origen = rng.choice(AEROPUERTOS, n, p=pesos)
    destino = rng.choice(AEROPUERTOS, n, p=pesos)
    aerolinea = rng.choice(AEROLINEAS, n)
    distancia = rng.integers(150, 2600, n).astype(float)
    duracion = np.round(30 + distancia / 7.5 + rng.normal(0, 12, n))
    salida_min = rng.integers(5 * 60, 23 * 60 + 30, n)            # entre 05:00 y 23:30
    llegada_min = (salida_min + duracion) % 1440
    crs_dep = (salida_min // 60) * 100 + salida_min % 60          # formato HHMM
    crs_arr = (llegada_min // 60) * 100 + llegada_min % 60

    # Probabilidad de retraso: tarde, hubs, mes, aerolínea, fin de semana + ruido.
    efecto_aerolinea = {a: e for a, e in zip(AEROLINEAS, rng.normal(0, 0.25, len(AEROLINEAS)))}
    logit = (-1.75 + 0.09 * (salida_min / 60 - 12)
             + 0.25 * pd.Series(origen).isin(HUBS).to_numpy()
             + 0.15 * pd.Series(destino).isin(HUBS).to_numpy()
             + EFECTO_MES[mes] + pd.Series(aerolinea).map(efecto_aerolinea).to_numpy()
             + 0.1 * (fecha.dt.dayofweek.to_numpy() >= 5) + rng.normal(0, 0.8, n))
    retraso = rng.random(n) < 1 / (1 + np.exp(-logit))
    arr_delay = np.where(retraso, rng.gamma(2, 25, n) + 15, rng.normal(-8, 7, n)).round()
    dep_delay = (arr_delay + rng.normal(4, 6, n)).round()          # columna de FUGA: casi igual

    cancelado = (rng.random(n) < 0.02).astype(float)
    desviado = ((rng.random(n) < 0.003) & (cancelado == 0)).astype(float)
    no_opera = (cancelado + desviado) > 0

    df = pd.DataFrame({
        "YEAR": anio, "MONTH": mes, "DAY_OF_MONTH": fecha.dt.day,
        "DAY_OF_WEEK": fecha.dt.dayofweek + 1,                     # el BTS usa 1 = lunes ... 7 = domingo
        # Como en los CSV reales, unos meses traen "6/30/2025 12:00:00 AM" y otros "06-01-25 00:00".
        "FL_DATE": ((fecha.dt.month.astype(str) + "/" + fecha.dt.day.astype(str) + "/"
                     + str(anio) + " 12:00:00 AM") if mes % 2 == 0
                    else fecha.dt.strftime("%m-%d-%y 00:00")),
        "OP_UNIQUE_CARRIER": aerolinea, "TAIL_NUM": [f"N{x}" for x in rng.integers(100, 999, n)],
        "OP_CARRIER_FL_NUM": rng.integers(1, 9999, n),
        "ORIGIN_AIRPORT_ID": 10000 + pd.Series(origen).astype("category").cat.codes.astype(int),
        "ORIGIN": origen,
        "DEST_AIRPORT_ID": 10000 + pd.Series(destino).astype("category").cat.codes.astype(int),
        "DEST": destino, "CRS_DEP_TIME": crs_dep, "DEP_TIME": crs_dep, "DEP_DELAY": dep_delay,
        "DEP_DEL15": (dep_delay >= 15).astype(float), "TAXI_OUT": rng.integers(8, 30, n),
        "WHEELS_OFF": crs_dep, "WHEELS_ON": crs_arr, "TAXI_IN": rng.integers(3, 15, n),
        "CRS_ARR_TIME": crs_arr, "ARR_TIME": crs_arr, "ARR_DELAY": arr_delay,
        "ARR_DEL15": (arr_delay >= 15).astype(float), "CANCELLED": cancelado, "DIVERTED": desviado,
        "CRS_ELAPSED_TIME": duracion, "ACTUAL_ELAPSED_TIME": duracion + arr_delay - dep_delay,
        "AIR_TIME": duracion - 20, "DISTANCE": distancia,
        "CARRIER_DELAY": np.where(retraso, arr_delay * 0.4, np.nan),
        "WEATHER_DELAY": np.where(retraso, 0, np.nan), "NAS_DELAY": np.where(retraso, arr_delay * 0.3, np.nan),
        "SECURITY_DELAY": np.where(retraso, 0, np.nan),
        "LATE_AIRCRAFT_DELAY": np.where(retraso, arr_delay * 0.3, np.nan),
    })
    # Como en los crudos: cancelados/desviados sin llegada, y algunas duraciones vacías.
    for col in ["ARR_DELAY", "ARR_DEL15", "ARR_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME"]:
        df.loc[no_opera, col] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "CRS_ELAPSED_TIME"] = np.nan
    return pd.concat([df, df.iloc[:2]], ignore_index=True)        # 2 duplicados a propósito


def generar(destino: Path, filas: int, semilla: int = 0) -> None:
    rng = np.random.default_rng(semilla)
    destino.mkdir(parents=True, exist_ok=True)
    for abrev, anio, mes in MESES:
        generar_mes(abrev, anio, mes, filas, rng).to_csv(destino / f"{abrev}-{anio}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destino", type=Path, required=True)
    parser.add_argument("--filas", type=int, default=20000)
    a = parser.parse_args()
    generar(a.destino, a.filas)
