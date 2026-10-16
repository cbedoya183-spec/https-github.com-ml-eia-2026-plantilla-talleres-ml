"""
experiment.py — PASO 5: comparar varios modelos con la MISMA vara.

Contiene:
- separar_temporal ......... entrenamiento = pasado, prueba = futuro (último 25 %)
- regla_manual ............. lo que haría alguien sin ML (punto de comparación)
- definir_modelos .......... línea base + 6 familias de modelos (incluye red neuronal)
- elegir_umbral ............ umbral que maximiza el F1 del retraso
- validar_modelo ........... validación cruzada temporal de un modelo
- comparar_modelos ......... la tabla con todos los modelos
- elegir_campeon ........... la regla de decisión (AUC y simplicidad)
- ajustar_hiperparametros .. búsqueda pequeña y honesta, solo en entrenamiento
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    COLUMNAS_MODELO,
    FRACCION_PRUEBA,
    GANANCIA_MINIMA,
    HUBS,
    MARGEN_SIMPLICIDAD,
    PLIEGUES_AJUSTE,
    PLIEGUES_CV,
    SEMILLA,
    TARGET,
)
from src.models.pipeline import crear_pipeline
from src.utils.logger import get_logger

logger = get_logger("experiment")


# -----------------------------------------------------------------------------
# 1. PARTICIÓN TEMPORAL
# -----------------------------------------------------------------------------
def separar_temporal(df: pd.DataFrame, fraccion_prueba: float = FRACCION_PRUEBA):
    """Separa por tiempo: el modelo se entrena con el pasado y se prueba con el futuro.

    Un corte aleatorio mezclaría vuelos de 2026 en el entrenamiento para predecir
    vuelos de 2025: eso no pasa en la vida real y daría resultados optimistas.
    `df` debe venir ordenado por fecha (lo hace limpiar_filas).
    """
    corte = int(len(df) * (1 - fraccion_prueba))          # posición donde empieza la prueba
    train, test = df.iloc[:corte], df.iloc[corte:]        # primeras 75 % filas / últimas 25 %
    X_train, y_train = train[COLUMNAS_MODELO], train[TARGET]
    X_test, y_test = test[COLUMNAS_MODELO], test[TARGET]
    logger.info(f"Entrenamiento: {len(train):,} vuelos ({train['FL_DATE'].min():%Y-%m-%d} a "
                f"{train['FL_DATE'].max():%Y-%m-%d}), retraso {y_train.mean():.2%}")
    logger.info(f"Prueba:        {len(test):,} vuelos ({test['FL_DATE'].min():%Y-%m-%d} a "
                f"{test['FL_DATE'].max():%Y-%m-%d}), retraso {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# 2. PUNTOS DE PARTIDA: regla manual y línea base
# -----------------------------------------------------------------------------
def regla_manual(X: pd.DataFrame) -> np.ndarray:
    """Regla 'de sentido común' sin ML:
    'Un vuelo se retrasa si sale desde las 16:00 y su origen o destino es un hub'.
    (Por la tarde ya se acumuló el efecto cascada y los hubs están congestionados.)"""
    tarde = X["CRS_DEP_HOUR"] >= 16                                  # sale a las 4 p. m. o después
    hub = X["ORIGIN"].isin(HUBS) | X["DEST"].isin(HUBS)               # toca un aeropuerto congestionado
    return (tarde & hub).astype(int).to_numpy()                       # 1 = predice retraso


# -----------------------------------------------------------------------------
# 3. LOS MODELOS (ordenados del más simple al más complejo)
# -----------------------------------------------------------------------------
def definir_modelos(y_train: pd.Series, rapido: bool = False) -> dict:
    """Devuelve {nombre: modelo}. `rapido=True` usa menos árboles (para ensayar en vivo)."""
    n_arboles = 60 if rapido else 200                                # número de árboles de los ensambles
    # Proporción puntuales/retrasados: XGBoost la usa para dar más peso a la clase minoritaria.
    peso_positivos = float((y_train == 0).sum() / (y_train == 1).sum())

    return {
        # Línea base: siempre predice la clase más frecuente ("puntual"). AUC = 0,5.
        "Línea base": DummyClassifier(strategy="most_frequent"),

        # Sesión 3 · Regresión logística. class_weight="balanced" compensa que solo
        # ~22 % de los vuelos se retrasan (si no, el modelo casi nunca diría "retraso").
        "Regresión logística": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=SEMILLA),

        # Sesión 4 · Árbol de decisión. Profundidad y hojas limitadas para no memorizar.
        "Árbol de decisión": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=100, class_weight="balanced", random_state=SEMILLA),

        # Sesión 5 · Random Forest: muchos árboles con muestras y columnas al azar (bagging).
        "Random Forest": RandomForestClassifier(
            n_estimators=n_arboles, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=SEMILLA),

        # Sesión 5 · Gradient boosting: cada árbol corrige los residuos del anterior.
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=n_arboles, learning_rate=0.05, max_leaf_nodes=31,
            class_weight="balanced", random_state=SEMILLA),

        # Sesión 7 · XGBoost: boosting con regularización (lambda, gamma).
        "XGBoost": XGBClassifier(
            n_estimators=n_arboles, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=peso_positivos, eval_metric="logloss",
            n_jobs=-1, random_state=SEMILLA),

        # Sesión 8 · Red neuronal (perceptrón multicapa, MLP).
        # Capa de entrada (≈90 columnas tras el one-hot) -> 32 neuronas -> 16 neuronas -> 1 salida.
        # - activation="relu": cada neurona calcula max(0, suma ponderada + sesgo) -> curvas, no solo rectas.
        # - La salida usa la sigmoide (logística) y da la probabilidad de retraso.
        # - Los pesos se aprenden con backpropagation + Adam (descenso de gradiente con impulso).
        # - alpha: penalización L2 sobre los pesos (como Ridge), evita memorizar.
        # - early_stopping: aparta el 10 % del entrenamiento y para cuando la pérdida en
        #   ese 10 % deja de mejorar 10 épocas seguidas (otra defensa contra el sobreajuste).
        # - El escalado (StandardScaler del Pipeline) es OBLIGATORIO: sin él el gradiente se desbalancea.
        # MLPClassifier no tiene class_weight: el desbalance se compensa con el UMBRAL elegido por F1.
        "Red neuronal (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=1e-3, learning_rate_init=1e-3, batch_size=256,
            max_iter=60 if rapido else 200, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10, random_state=SEMILLA),
    }


# Orden de simplicidad (menor = más fácil de explicar y de entrenar).
SIMPLICIDAD = {"Línea base": 0, "Regresión logística": 1, "Árbol de decisión": 2,
               "Random Forest": 3, "Gradient boosting": 4, "XGBoost": 5,
               "Red neuronal (MLP)": 6}   # la red es la menos interpretable (caja negra)


# -----------------------------------------------------------------------------
# 4. UMBRAL DE DECISIÓN
# -----------------------------------------------------------------------------
def elegir_umbral(y_real, probabilidad) -> tuple[float, float]:
    """Prueba umbrales de 0,05 a 0,95 y devuelve el que da el mayor F1 del retraso.

    predict() corta en 0,50 sin avisar. Con class_weight='balanced' las
    probabilidades se corren hacia arriba, así que 0,50 no es el mejor punto.
    """
    mejor_umbral, mejor_f1 = 0.5, 0.0      # si ningún umbral detecta retrasos (línea base), queda 0,5
    for umbral in np.round(np.arange(0.05, 0.96, 0.05), 2):         # 0.05, 0.10, ..., 0.95
        prediccion = (probabilidad >= umbral).astype(int)             # 1 si supera el umbral
        f1 = f1_score(y_real, prediccion, zero_division=0)            # F1 de la clase "retraso"
        if f1 > mejor_f1:
            mejor_umbral, mejor_f1 = float(umbral), float(f1)
    return mejor_umbral, mejor_f1


# -----------------------------------------------------------------------------
# 5. VALIDACIÓN CRUZADA TEMPORAL (la misma para todos los modelos)
# -----------------------------------------------------------------------------
def validar_modelo(modelo, X: pd.DataFrame, y: pd.Series, n_pliegues: int = PLIEGUES_CV) -> dict:
    """Valida un modelo con TimeSeriesSplit y devuelve AUC promedio, desviación y umbral.

    TimeSeriesSplit con 3 pliegues sobre el entrenamiento:
        pliegue 1: entrena [====]            valida [==]
        pliegue 2: entrena [======]          valida [==]
        pliegue 3: entrena [========]        valida [==]
    Siempre se entrena con el pasado y se valida con lo que sigue.
    """
    cv = TimeSeriesSplit(n_splits=n_pliegues)
    aucs = []
    inicio = time.time()
    for idx_train, idx_val in cv.split(X):                            # recorre los 3 pliegues
        pipe = crear_pipeline(clone(modelo))                          # modelo "nuevo" en cada pliegue
        pipe.fit(X.iloc[idx_train], y.iloc[idx_train])                # aprende solo con el pasado
        prob = pipe.predict_proba(X.iloc[idx_val])[:, 1]              # probabilidad de retraso
        aucs.append(roc_auc_score(y.iloc[idx_val], prob))             # qué tan bien ordena
    # El umbral se elige en el ÚLTIMO pliegue (el más parecido al futuro).
    umbral, f1_cv = elegir_umbral(y.iloc[idx_val], prob)
    return {
        "roc_auc_cv": float(np.mean(aucs)),        # promedio de los 3 pliegues
        "roc_auc_cv_std": float(np.std(aucs)),     # qué tanto varía entre pliegues
        "umbral": umbral,                          # punto de operación elegido
        "f1_cv_umbral": f1_cv,                     # F1 del último pliegue en ese umbral
        "segundos_cv": round(time.time() - inicio, 1),
    }


def comparar_modelos(modelos: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Valida todos los modelos con los mismos pliegues y devuelve la tabla ordenada."""
    filas = []
    for nombre, modelo in modelos.items():
        logger.info(f"Validando {nombre} ...")
        resultado = validar_modelo(modelo, X, y)
        filas.append({"modelo": nombre, **resultado})
        logger.info(f"  ROC-AUC = {resultado['roc_auc_cv']:.4f} ± {resultado['roc_auc_cv_std']:.4f}"
                    f" | umbral {resultado['umbral']:.2f} | {resultado['segundos_cv']} s")
    return pd.DataFrame(filas).sort_values("roc_auc_cv", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 6. ELECCIÓN DEL MODELO
# -----------------------------------------------------------------------------
def elegir_campeon(tabla_cv: pd.DataFrame) -> tuple[str, str]:
    """Regla: gana el mayor ROC-AUC de validación; pero si un modelo más simple
    queda a menos de 0,01, se prefiere el simple (como Ridge vs XGBoost en las casas)."""
    candidatos = tabla_cv[tabla_cv["modelo"] != "Línea base"]
    mejor = candidatos.iloc[0]                                       # el de mayor AUC
    cercanos = candidatos[candidatos["roc_auc_cv"] >= mejor["roc_auc_cv"] - MARGEN_SIMPLICIDAD]
    # Entre los "empatados", el de menor número en SIMPLICIDAD.
    campeon = min(cercanos["modelo"], key=lambda m: SIMPLICIDAD[m])
    auc_campeon = float(candidatos.loc[candidatos["modelo"] == campeon, "roc_auc_cv"].iloc[0])

    if campeon == mejor["modelo"]:
        texto = (f"{campeon} tiene el mayor ROC-AUC de validación ({auc_campeon:.4f}) "
                 f"y ningún modelo más simple queda a menos de {MARGEN_SIMPLICIDAD}.")
    else:
        texto = (f"{mejor['modelo']} tiene el mayor ROC-AUC de validación ({mejor['roc_auc_cv']:.4f}), "
                 f"pero {campeon} queda a {mejor['roc_auc_cv'] - auc_campeon:.4f} "
                 f"(dentro del margen de {MARGEN_SIMPLICIDAD}) y es más simple: se elige {campeon}.")
    return campeon, texto


# -----------------------------------------------------------------------------
# 7. HIPERPARÁMETROS (búsqueda pequeña: 4 combinaciones por modelo)
# -----------------------------------------------------------------------------
# Las claves llevan el prefijo "modelo__" porque el modelo es el paso "modelo" del Pipeline.
REJILLAS = {
    "Regresión logística": {"modelo__C": [0.05, 0.3, 1.0, 4.0]},           # fuerza de regularización
    "Árbol de decisión": {"modelo__max_depth": [6, 10], "modelo__min_samples_leaf": [50, 200]},
    "Random Forest": {"modelo__max_depth": [8, 14], "modelo__min_samples_leaf": [10, 40]},
    "Gradient boosting": {"modelo__learning_rate": [0.05, 0.1], "modelo__max_leaf_nodes": [15, 31]},
    "XGBoost": {"modelo__max_depth": [4, 6], "modelo__learning_rate": [0.05, 0.1]},
    "Red neuronal (MLP)": {"modelo__hidden_layer_sizes": [(32,), (32, 16)],   # 1 o 2 capas ocultas
                           "modelo__alpha": [1e-4, 1e-2]},                    # regularización L2
}


def ajustar_hiperparametros(nombre: str, modelo, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compara la configuración base contra 4 combinaciones, con TimeSeriesSplit(2)
    SOLO sobre el entrenamiento. Se adopta lo nuevo solo si gana al menos 0,005 de AUC."""
    cv = TimeSeriesSplit(n_splits=PLIEGUES_AJUSTE)
    pipe = crear_pipeline(clone(modelo))

    # AUC de la configuración base, medido con los MISMOS pliegues que la búsqueda.
    auc_base = float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())

    busqueda = GridSearchCV(pipe, REJILLAS[nombre], cv=cv, scoring="roc_auc", n_jobs=1)
    busqueda.fit(X, y)                                               # prueba las 4 combinaciones
    ganancia = busqueda.best_score_ - auc_base
    return {
        "modelo": nombre,
        "roc_auc_base": round(auc_base, 4),
        "roc_auc_ajustado": round(float(busqueda.best_score_), 4),
        "ganancia": round(float(ganancia), 4),
        "mejores_parametros": {k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items()},
        "se_adopta": bool(ganancia >= GANANCIA_MINIMA),
        "estimador": busqueda.best_estimator_.named_steps["modelo"],  # modelo con los mejores parámetros
    }
