"""
train.py — La misma corrida del notebook, pero desde la terminal (pasos 3 a 6).

Uso (en la raíz del repositorio, con la muestra ya creada):
    python src/models/train.py            # corrida completa (la oficial)
    python src/models/train.py --rapido   # menos árboles: ensayo rápido para la exposición

Escribe en models/:
    metrics_comparison.csv  tabla final (validación + prueba) de todos los modelos
    hiperparametros.csv     resultado de la búsqueda de hiperparámetros
    eleccion.txt            el argumento escrito de por qué se eligió el modelo
    best_model.joblib       Pipeline entrenado + umbral (no se sube a Git)
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

RAIZ = Path(__file__).resolve().parents[2]          # raíz del repo, para poder importar src
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import MODELS_DIR, REPORTS_DIR  # noqa: E402
from src.data.load_data import cargar_muestra  # noqa: E402
from src.features.build_features import preparar_tabla  # noqa: E402
from src.features.imputation import comparar_imputadores  # noqa: E402
from src.models.evaluate import metricas  # noqa: E402
from src.models.experiment import (  # noqa: E402
    ajustar_hiperparametros,
    comparar_modelos,
    definir_modelos,
    elegir_campeon,
    regla_manual,
    separar_temporal,
    validar_modelo,
)
from src.models.pipeline import crear_pipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("train")


def correr_todo(rapido: bool = False) -> dict:
    """Ejecuta los pasos 3 a 6 y guarda las tablas de evidencia."""
    # ---- Pasos 3 y 4: limpieza y columnas --------------------------------------
    tabla = preparar_tabla(cargar_muestra())
    X_train, X_test, y_train, y_test = separar_temporal(tabla)

    # Comparación de imputadores para la duración programada (solo entrenamiento).
    imputacion = comparar_imputadores(X_train["CRS_ELAPSED_TIME"], X_train["OP_UNIQUE_CARRIER"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    imputacion.to_csv(REPORTS_DIR / "comparacion_imputacion.csv", index=False)
    logger.info("Imputación de CRS_ELAPSED_TIME (MAE en minutos):\n" + imputacion.to_string(index=False))

    # ---- Paso 5: comparación con la misma validación ----------------------------
    modelos = definir_modelos(y_train, rapido=rapido)
    tabla_cv = comparar_modelos(modelos, X_train, y_train)
    campeon, argumento = elegir_campeon(tabla_cv)
    logger.info(argumento)

    # ---- Hiperparámetros: el campeón y el de mayor AUC ---------------------------
    candidatos = [campeon]
    mejor_auc = tabla_cv[tabla_cv["modelo"] != "Línea base"].iloc[0]["modelo"]
    if mejor_auc != campeon:
        candidatos.append(mejor_auc)
    ajustes = [ajustar_hiperparametros(n, modelos[n], X_train, y_train) for n in candidatos]
    for a in ajustes:
        logger.info(f"Ajuste {a['modelo']}: base {a['roc_auc_base']} -> {a['roc_auc_ajustado']} "
                    f"(ganancia {a['ganancia']:+.4f}) -> {'SE ADOPTA' if a['se_adopta'] else 'no se adopta'}")
        if a["se_adopta"]:                                  # solo si gana >= 0,005
            modelos[a["modelo"]] = a["estimador"]
            nuevo = validar_modelo(a["estimador"], X_train, y_train)   # nuevo umbral con la misma CV
            for k, v in nuevo.items():
                tabla_cv.loc[tabla_cv["modelo"] == a["modelo"], k] = v
    if any(a["se_adopta"] for a in ajustes):               # la elección se rehace con los AUC nuevos
        tabla_cv = tabla_cv.sort_values("roc_auc_cv", ascending=False).reset_index(drop=True)
        campeon, argumento = elegir_campeon(tabla_cv)

    # ---- Paso 6: prueba final (cada modelo con SU umbral ya fijado) ----------------
    filas, pipelines, probabilidades = [], {}, {}
    for nombre, modelo in modelos.items():
        pipe = crear_pipeline(clone(modelo)).fit(X_train, y_train)     # entrena con TODO el entrenamiento
        umbral = float(tabla_cv.loc[tabla_cv["modelo"] == nombre, "umbral"].iloc[0])
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= umbral).astype(int)
        fila_cv = tabla_cv[tabla_cv["modelo"] == nombre].iloc[0]
        filas.append({"modelo": nombre, "roc_auc_cv": round(fila_cv["roc_auc_cv"], 4),
                      "roc_auc_cv_std": round(fila_cv["roc_auc_cv_std"], 4), "umbral": umbral,
                      **metricas(y_test, pred, prob)})
        pipelines[nombre], probabilidades[nombre] = pipe, prob
    filas.append({"modelo": "Regla manual", "roc_auc_cv": None, "roc_auc_cv_std": None, "umbral": None,
                  **metricas(y_test, regla_manual(X_test))})
    resultados = pd.DataFrame(filas)

    # ---- Guardar evidencia ---------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(MODELS_DIR / "metrics_comparison.csv", index=False)
    pd.DataFrame([{k: v for k, v in a.items() if k != "estimador"} for a in ajustes]) \
        .to_csv(MODELS_DIR / "hiperparametros.csv", index=False)
    umbral_campeon = float(tabla_cv.loc[tabla_cv["modelo"] == campeon, "umbral"].iloc[0])
    joblib.dump({"pipeline": pipelines[campeon], "umbral": umbral_campeon, "modelo": campeon},
                MODELS_DIR / "best_model.joblib")
    fila = resultados[resultados["modelo"] == campeon].iloc[0]
    regla = resultados[resultados["modelo"] == "Regla manual"].iloc[0]
    (MODELS_DIR / "eleccion.txt").write_text(
        f"Modelo elegido: {campeon} (umbral {umbral_campeon:.2f})\n\n{argumento}\n\n"
        f"En la prueba (no usada para elegir): F1 {fila['f1_retraso']:.4f}, "
        f"recall {fila['recall_retraso']:.4f}, precisión {fila['precision_retraso']:.4f}, "
        f"ROC-AUC {fila['roc_auc']:.4f}.\n"
        f"Regla manual en la prueba: F1 {regla['f1_retraso']:.4f}.\n", encoding="utf-8")

    logger.info("Resultados finales:\n" + resultados.to_string(index=False))
    return {"tabla": tabla, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "tabla_cv": tabla_cv, "resultados": resultados, "campeon": campeon,
            "pipelines": pipelines, "probabilidades": probabilidades}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparación de modelos de retraso de vuelos")
    parser.add_argument("--rapido", action="store_true", help="menos árboles (ensayo rápido)")
    correr_todo(rapido=parser.parse_args().rapido)
