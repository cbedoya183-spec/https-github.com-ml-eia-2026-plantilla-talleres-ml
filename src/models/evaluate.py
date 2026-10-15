"""
evaluate.py — PASO 6: métricas en la prueba y gráficas para la exposición.

Métricas usadas (clasificación desbalanceada, ~22 % de retrasos):
- ROC-AUC: qué tan bien ORDENA el modelo (¿los retrasados reciben más probabilidad?).
           No depende del umbral. 0,5 = azar, 1,0 = perfecto.
- PR-AUC:  igual, pero centrado en la clase minoritaria (retraso). Su azar = tasa de retraso.
- Precisión: de los vuelos que el modelo marca como "retraso", cuántos lo fueron.
- Recall:    de los vuelos que se retrasaron, cuántos detectó el modelo.
- F1:        media armónica de precisión y recall (equilibrio entre ambas).
- Exactitud: % de aciertos. ENGAÑA aquí: decir siempre "puntual" ya acierta ~78 %.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def metricas(y_real, prediccion, probabilidad=None) -> dict:
    """Calcula las métricas de clasificación. Sin probabilidad no hay AUC (regla manual)."""
    resultado = {
        "exactitud": accuracy_score(y_real, prediccion),
        "precision_retraso": precision_score(y_real, prediccion, zero_division=0),
        "recall_retraso": recall_score(y_real, prediccion, zero_division=0),
        "f1_retraso": f1_score(y_real, prediccion, zero_division=0),
        "roc_auc": np.nan,                                   # se llena solo si hay probabilidades
        "pr_auc": np.nan,
    }
    if probabilidad is not None:
        resultado["roc_auc"] = roc_auc_score(y_real, probabilidad)
        resultado["pr_auc"] = average_precision_score(y_real, probabilidad)
    return {k: round(float(v), 4) for k, v in resultado.items()}


def graficar_comparacion_cv(tabla_cv: pd.DataFrame, ruta=None):
    """Barras horizontales con el ROC-AUC de validación (y su desviación) de cada modelo."""
    tabla = tabla_cv.sort_values("roc_auc_cv")                       # el mejor queda arriba
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(tabla["modelo"], tabla["roc_auc_cv"], xerr=tabla["roc_auc_cv_std"],
            color="#4C72B0", capsize=4)                              # barra = promedio, línea = ± desviación
    ax.axvline(0.5, color="grey", linestyle="--", label="Azar (0,5)")
    ax.set_xlim(0.45, max(0.75, tabla["roc_auc_cv"].max() + 0.05))
    ax.set_xlabel("ROC-AUC promedio en validación cruzada temporal (3 pliegues)")
    ax.set_title("Paso 5 · Comparación de modelos con la misma validación")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=120)
    return fig


def graficar_curvas(y_test, probabilidades: dict, ruta=None):
    """Curva ROC y curva Precisión-Recall de varios modelos en la prueba."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for nombre, prob in probabilidades.items():
        fpr, tpr, _ = roc_curve(y_test, prob)                        # tasas de falsos y verdaderos positivos
        ax1.plot(fpr, tpr, label=f"{nombre} ({roc_auc_score(y_test, prob):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ax2.plot(rec, prec, label=f"{nombre} ({average_precision_score(y_test, prob):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", label="Azar")                    # diagonal = modelo aleatorio
    ax1.set(xlabel="Tasa de falsos positivos", ylabel="Recall (verdaderos positivos)", title="Curva ROC")
    ax2.axhline(np.mean(y_test), color="k", linestyle="--", label="Azar (tasa de retraso)")
    ax2.set(xlabel="Recall", ylabel="Precisión", title="Curva Precisión-Recall")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=120)
    return fig


def graficar_matrices(y_test, predicciones: dict, ruta=None):
    """Una matriz de confusión por cada predicción (modelo elegido, regla manual, etc.)."""
    fig, ejes = plt.subplots(1, len(predicciones), figsize=(5 * len(predicciones), 4))
    ejes = np.atleast_1d(ejes)                                       # funciona con 1 o varias matrices
    for ax, (nombre, pred) in zip(ejes, predicciones.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Puntual", "Retraso"], cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(nombre)
        ax.set_xlabel("Predicción del modelo")
        ax.set_ylabel("Lo que pasó realmente")
        ax.grid(False)                                               # sin cuadrícula encima de los números
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=120)
    return fig


def importancia_columnas(pipeline, n: int = 15) -> pd.DataFrame:
    """Columnas más influyentes del modelo final.
    - Logística: coeficiente (positivo = sube la probabilidad de retraso).
    - Árboles/ensambles: feature_importances_ (cuánto ayuda cada columna a separar)."""
    nombres = pipeline.named_steps["preprocesamiento"].get_feature_names_out()  # columnas tras one-hot
    modelo = pipeline.named_steps["modelo"]
    if hasattr(modelo, "coef_"):
        valores = modelo.coef_[0]
        tabla = pd.DataFrame({"columna": nombres, "coeficiente": valores,
                              "odds_ratio": np.exp(valores)})       # e^coef: cuánto multiplica los odds
        orden = tabla["coeficiente"].abs().sort_values(ascending=False).index
    elif hasattr(modelo, "feature_importances_"):
        tabla = pd.DataFrame({"columna": nombres, "importancia": modelo.feature_importances_})
        orden = tabla["importancia"].sort_values(ascending=False).index
    else:
        return pd.DataFrame()
    return tabla.loc[orden].head(n).reset_index(drop=True)
