# Diccionario de datos

**Fuente:** BTS / U.S. DOT, *Reporting Carrier On-Time Performance* · **Descarga:** [Kaggle — 2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights) · **Licencia:** dominio público.

## Etiqueta
| Columna | Descripción | Rol |
|---|---|---|
| `ARR_DEL15` | 1 si la llegada fue ≥ 15 min después de lo programado (FAA/DOT) | **Variable objetivo** |

## Columnas que entran al modelo (todas conocidas antes del vuelo)
| Columna | Tipo | Origen | Tratamiento en el Pipeline |
|---|---|---|---|
| `OP_UNIQUE_CARRIER` | categórica | original | one-hot |
| `ORIGIN`, `DEST` | categórica | original | one-hot de los 25 más frecuentes + "infrequent" |
| `CRS_ELAPSED_TIME` | numérica | original | mediana **por aerolínea** (`GroupMedianImputer`) + escalado |
| `DISTANCE` | numérica | original | mediana + escalado |
| `MONTH`, `DAY_OF_MONTH` | numérica | de `FL_DATE` | mediana + escalado |
| `DAY_OF_WEEK` (0=lunes), `IS_WEEKEND` | numérica | creada de `FL_DATE` | mediana + escalado |
| `CRS_DEP_HOUR`, `CRS_ARR_HOUR` | numérica | creada de `CRS_*_TIME` (HHMM; 2400 → 0) | mediana + escalado |
| `CRS_DEP_SIN/COS`, `MONTH_SIN/COS` | numérica | creada (codificación cíclica) | escalado |
| `ORIGIN_IS_HUB`, `DEST_IS_HUB`, `IS_HUB_TO_HUB` | 0/1 | creada (10 hubs FAA) | escalado |
| `DEP_TIME_BLOCK` | categórica | creada (madrugada/mañana/tarde/noche) | one-hot |
| `DISTANCE_CAT` | categórica | creada (corta/media/larga) | one-hot |
| `SEASON` | categórica | creada del mes | one-hot |

## Columnas descartadas
| Columnas | Motivo |
|---|---|
| `DEP_TIME`, `DEP_DELAY`, `DEP_DEL15`, `TAXI_OUT` | **Fuga**: se conocen al salir de puerta |
| `WHEELS_OFF`, `WHEELS_ON`, `AIR_TIME`, `ACTUAL_ELAPSED_TIME` | **Fuga**: se conocen en el aire |
| `TAXI_IN`, `ARR_TIME`, `ARR_DELAY` | **Fuga**: definen el resultado |
| `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY` | **Fuga**: causas calculadas después del vuelo |
| `TAIL_NUM`, `OP_CARRIER_FL_NUM` | identificadores que no generalizan |
| `ORIGIN_AIRPORT_ID`, `DEST_AIRPORT_ID` | redundantes con `ORIGIN`/`DEST` |
| `CANCELLED`, `DIVERTED` | solo sirven para filtrar (vuelos sin etiqueta) |
