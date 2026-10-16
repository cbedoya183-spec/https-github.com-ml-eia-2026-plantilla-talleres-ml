# ✈️ Predicción de retrasos en vuelos comerciales de EE.UU. (jun-2025 a may-2026)
### Proyecto final · Machine Learning · Especialización en Analítica y Ciencia de Datos · Universidad EIA
**Rama:** `CARLOS-BEDOYA` · **Autor de esta versión:** Carlos Bedoya

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)
![Fuente](https://img.shields.io/badge/fuente-BTS%20%2F%20DOT-blue.svg)

> **Pregunta:** con lo que se sabe **antes de que el avión salga** (itinerario, ruta, aerolínea, fecha), ¿llegará el vuelo **15 minutos o más tarde**?
> Clasificación binaria supervisada, con la etiqueta oficial de la FAA/DOT (`ARR_DEL15`).

Esta rama resuelve el proyecto completo siguiendo **los 6 pasos de la sesión 6** que pide la rúbrica, con el mismo planteamiento del equipo, y con **código comentado línea por línea** para poder explicar cualquier parte en la exposición.

---

## 1 · ¿Por qué Machine Learning?
El retraso depende de factores que se combinan de forma no lineal: la **hora** (un avión hace varios trayectos y el retraso de la mañana se propaga a la tarde), la **temporada** (tormentas en verano, nieve en invierno, Navidad, *Spring Break*), el **aeropuerto** (congestión de los *hubs*) y la **aerolínea**.

**Regla a mano (punto de comparación):** *“se retrasa si sale a las 16:00 o después y su origen o destino es uno de los 10 hubs (ATL, ORD, DFW, DEN, CLT, LAX, JFK, MIA, MCO, EWR)”*.
**Línea base:** decir siempre “puntual” (acierta ~78 % pero no detecta ningún retraso).

## 2 · Datos
| | |
|---|---|
| Publica | *Bureau of Transportation Statistics (BTS)*, U.S. DOT — *Reporting Carrier On-Time Performance* |
| Descarga | Kaggle: [2021-2026 US Flights](https://www.kaggle.com/datasets/umerhafeezsiddiqui/2021-2026-us-flights), un CSV por mes (API o manual) |
| Licencia | Obra del Gobierno Federal de EE.UU. (dominio público) |
| Periodo | 1-jun-2025 a 31-may-2026 (12 meses: un ciclo estacional completo) |
| Una fila | Un vuelo comercial doméstico programado |
| Tamaño | 12 CSV crudos, **7.786.702 filas**, ~80 MB por mes |
| Tabla de modelado | **1 % de los vuelos operados de cada mes**, al azar, semilla 42 → **~76.200 filas** |

## 3 · Limpieza (auditoría real de los 12 CSV, `reports/auditoria_mensual_cruda.csv`)
| Concepto | Filas |
|---|---:|
| Universo crudo | 7.786.702 |
| Cancelados | 145.007 |
| Desviados | 21.637 |
| Sin etiqueta `ARR_DEL15` | 166.645 |
| Operados con etiqueta | 7.620.057 |
| Duplicados de la clave del vuelo | 3 |
| Duración programada vacía | 4 |

- Cancelados y desviados no llegan en su itinerario → **sin etiqueta → se excluyen** (no se imputa la etiqueta).
- `FL_DATE` llega en **dos formatos mezclados** (`6/30/2025 12:00:00 AM` y `06-01-25 00:00`) → `convertir_fecha()` los unifica.
- Imputación de `CRS_ELAPSED_TIME`: se **esconde el 20 %** del entrenamiento y se compara el error. En la corrida real del equipo la **mediana por aerolínea** bajó el MAE de 52,78 a 46,84 min (−11,2 %) → va **dentro del Pipeline** (`GroupMedianImputer`).
- **Fugas eliminadas** (se conocen después del despegue): `DEP_TIME`, `DEP_DELAY`, `DEP_DEL15`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `AIR_TIME`, `ACTUAL_ELAPSED_TIME`, `TAXI_IN`, `ARR_TIME`, `ARR_DELAY` y las 5 causas de retraso. El notebook demuestra que **solo con `DEP_DEL15` el AUC se dispara** a más de 0,9.

## 4 · Columnas
Entran 8 columnas originales del itinerario y se **crean 13**: hora de salida/llegada, franja horaria, seno/coseno de la hora y del mes (23 h junto a 0 h; diciembre junto a enero), día de la semana, fin de semana, estación, banderas de hub y categoría de distancia. Todo lo que *aprende* (imputar, escalar, one-hot con los 25 aeropuertos más frecuentes) vive **dentro del `Pipeline`**.

## 5 · Modelos y validación
- **Partición temporal:** entrenamiento = 75 % más antiguo (jun-2025 → mar-2026); **prueba = 25 % más reciente** (mar → may-2026), que no elige nada.
- **Validación:** `TimeSeriesSplit` de 3 pliegues dentro del entrenamiento, **igual para todos**. Métrica de elección: **ROC-AUC**.
- **Umbral:** se elige en el último pliegue maximizando el **F1 del retraso** (con `class_weight="balanced"` el 0,50 no es el punto de operación).
- **Familias:** línea base, regla manual, regresión logística (S3), árbol (S4), Random Forest y gradient boosting (S5), XGBoost (S7) y **red neuronal MLP (S8)**. KNN/SVM-kernel no entran por costo de predicción con ~57 mil filas en one-hot.
- **Red neuronal (S8):** perceptrón multicapa `entrada (≈90) → 32 → 16 → 1`, activación ReLU, salida sigmoide, entrenado con *backpropagation* + Adam; regularización L2 (`alpha`) y parada temprana. Usa las columnas escaladas del Pipeline y, como no admite `class_weight`, el desbalance se compensa con el umbral elegido por F1. El notebook muestra su número de parámetros y su curva de aprendizaje.
- **Regla de elección:** mayor ROC-AUC de validación; si un modelo más simple queda a menos de **0,01**, gana el simple. Hiperparámetros: 4 combinaciones, `TimeSeriesSplit(2)`, se adoptan solo si ganan **≥ 0,005**.

## 6 · Resultados de referencia (corrida del equipo con los datos reales)
Estos números son de la corrida del equipo con los 12 CSV reales (rama `Sebastian/Planteamiento`), con la **misma metodología** que implementa esta rama. La red neuronal es un modelo adicional de esta versión: su fila se llena al ejecutar el notebook. Al ejecutar el notebook con los CSV en `data/raw/` se regeneran en `models/metrics_comparison.csv`.

| Modelo | ROC-AUC validación | Umbral | Precisión | Recall | F1 retraso | ROC-AUC prueba |
|---|---:|---:|---:|---:|---:|---:|
| Regla manual | — | — | 0,3012 | 0,2608 | 0,2796 | — |
| Línea base | 0,5000 | 0,50 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |
| **Regresión logística** ✅ | **0,6193** | **0,30** | **0,2721** | **0,8073** | **0,4070** | **0,6524** |
| Árbol de decisión | 0,6107 | 0,40 | 0,2602 | 0,8244 | 0,3956 | 0,6207 |
| Random Forest | 0,6259 | 0,40 | 0,2713 | 0,7880 | 0,4037 | 0,6399 |
| Gradient boosting | 0,6273 | 0,40 | 0,2709 | 0,7923 | 0,4038 | 0,6374 |
| XGBoost | 0,6285 | 0,40 | 0,2661 | 0,8271 | 0,4026 | 0,6374 |
| Red neuronal (MLP) ¹ | — | — | — | — | — | — |

¹ Modelo agregado en esta versión: sus números se generan al ejecutar el notebook con los CSV reales.

**Elección argumentada:** XGBoost tiene el mayor AUC de validación (0,6285), pero la logística queda a 0,0092 (dentro del margen de 0,01, y menor que la desviación entre pliegues, 0,0253) y es **más simple, rápida e interpretable** → se elige la **regresión logística**. En la prueba detecta el **81 % de los retrasos** y su F1 (0,407) supera claramente a la regla manual (0,280). El ajuste de hiperparámetros no ganó ≥ 0,005, así que se conserva la configuración base.

> Los números exactos de esta rama pueden diferir en el tercer decimal de los del equipo, porque el sorteo del 1 % se implementa con `DataFrame.sample` (semilla 42) y la rejilla de hiperparámetros es fija. La conclusión debe confirmarse al correr con los CSV reales.

---

## 🏗️ Estructura
```text
├── README.md                 este documento
├── requirements.txt          dependencias
├── setup.py / pyproject.toml instala la carpeta src como paquete (pip install -e .)
├── data/
│   ├── raw/                  12 CSV de Kaggle (NO se suben a Git)
│   └── processed/            muestra del 1 % (se genera; NO se sube)
├── docs/
│   ├── diccionario_datos.md  columnas, decisiones y fugas
│   ├── guion_exposicion.md   qué decir y qué correr en la sesión 9
│   └── guia_colaboracion.md  flujo de Git con ramas
├── models/                   tablas de resultados y modelo (se generan al correr)
├── notebooks/
│   └── proyecto_final_vuelos_us.ipynb   los 6 pasos, explicado celda por celda
├── reports/                  auditoría de los CSV crudos y figuras
├── src/
│   ├── config.py             rutas, semilla, periodo y listas de columnas
│   ├── data/                 download_data.py (descarga + auditoría + muestra), load_data.py
│   ├── features/             build_features.py, transformers.py, imputation.py
│   ├── models/               pipeline.py, experiment.py, evaluate.py, train.py
│   └── utils/logger.py
└── tests/                    prueba de humo con datos simulados (python -m pytest)
```

## ▶️ Cómo correrlo
```bash
# 1. Clonar y entrar a esta rama
git clone https://github.com/sebastia-olaya/Proyecto-final-Machine-Learning.git
cd Proyecto-final-Machine-Learning
git checkout CARLOS-BEDOYA

# 2. Entorno e instalación
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# 3. Datos: poner los 12 CSV (Jun-2025.csv ... May-2026.csv) en data/raw/
#    o descargarlos con el token de Kaggle (~/.kaggle/kaggle.json):
python src/data/download_data.py   # descarga si faltan + auditoría + muestra del 1 %

# 4a. Notebook (la corrida oficial, ~3 minutos)
jupyter notebook notebooks/proyecto_final_vuelos_us.ipynb      # Kernel → Restart & Run All

# 4b. O lo mismo desde la terminal
python src/models/train.py             # completa
python src/models/train.py --rapido    # menos árboles: ensayo para la exposición

# 5. Prueba de humo (no necesita los datos reales)
python -m pytest -q
```
Semilla fija **42** en todo el proyecto: los números coinciden en cada corrida.
