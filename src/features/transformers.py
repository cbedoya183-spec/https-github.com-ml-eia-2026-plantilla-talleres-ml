"""
transformers.py — Transformador propio compatible con scikit-learn.

GroupMedianImputer rellena la duración programada (CRS_ELAPSED_TIME) con la
MEDIANA DE SU AEROLÍNEA. ¿Por qué un transformador y no un simple fillna()?
Porque así vive dentro del Pipeline: en cada pliegue de la validación cruzada
las medianas se calculan SOLO con las filas de entrenamiento de ese pliegue.
Si se calcularan con toda la tabla, la prueba "contaminaría" el entrenamiento.

Para que scikit-learn lo acepte basta con heredar de BaseEstimator y
TransformerMixin, y definir fit() (aprende) y transform() (aplica).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Imputa una columna numérica con la mediana del grupo al que pertenece la fila."""

    def __init__(self, columna: str = "CRS_ELAPSED_TIME", grupo: str = "OP_UNIQUE_CARRIER"):
        self.columna = columna    # columna que tiene vacíos (duración programada)
        self.grupo = grupo        # columna que define los grupos (aerolínea)

    def fit(self, X: pd.DataFrame, y=None):
        """APRENDE: la mediana de cada aerolínea y la mediana global (solo con entrenamiento)."""
        self.medianas_ = X.groupby(self.grupo)[self.columna].median()   # una mediana por aerolínea
        self.mediana_global_ = float(X[self.columna].median())          # respaldo para aerolíneas nuevas
        return self                                                     # convención de scikit-learn

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """APLICA: reemplaza cada vacío por la mediana aprendida de su aerolínea."""
        X = X.copy()
        # map() busca la mediana de la aerolínea de cada fila; si la aerolínea no
        # estaba en el entrenamiento, queda vacío y se usa la mediana global.
        mediana_fila = X[self.grupo].map(self.medianas_).astype(float).fillna(self.mediana_global_)
        X[self.columna] = X[self.columna].astype(float).fillna(mediana_fila)  # solo toca los vacíos
        return X

    def get_feature_names_out(self, input_features=None):
        """Las columnas de salida son las mismas de entrada (no crea ni quita columnas)."""
        return np.asarray(input_features) if input_features is not None else None
