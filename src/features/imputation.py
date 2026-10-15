"""
imputation.py — PASO 3: ¿con qué se rellenan los vacíos? Se compara, no se supone.

Como en el ejemplo de la edad del Titanic: se ESCONDE a propósito una parte de
los valores que sí conocemos, se rellenan con cada método y se mide el error
(MAE) contra el valor verdadero. Gana el método con menor error.

Se hace SOLO con el entrenamiento: la prueba no participa en ninguna decisión.
"""

import numpy as np
import pandas as pd

from src.config import SEMILLA


def comparar_imputadores(valores: pd.Series, grupos: pd.Series, fraccion: float = 0.20) -> pd.DataFrame:
    """Esconde el 20 % de `valores` y compara media global, mediana global y mediana por grupo."""
    rng = np.random.RandomState(SEMILLA)                     # generador aleatorio con semilla fija
    conocidos = valores.dropna()                             # solo filas donde sabemos la respuesta
    n_ocultos = int(len(conocidos) * fraccion)               # cuántos valores se esconden
    idx_ocultos = rng.choice(conocidos.index, size=n_ocultos, replace=False)

    verdad = conocidos.loc[idx_ocultos]                      # lo que "debería" salir
    visibles = conocidos.drop(index=idx_ocultos)             # lo que el método puede ver

    # Cada método produce una predicción para los valores escondidos.
    pred_media = np.full(n_ocultos, visibles.mean())         # 1) media de todos los vuelos
    pred_mediana = np.full(n_ocultos, visibles.median())     # 2) mediana de todos los vuelos
    mediana_grupo = visibles.groupby(grupos.loc[visibles.index]).median()      # 3) mediana por aerolínea
    pred_grupo = grupos.loc[idx_ocultos].map(mediana_grupo).fillna(visibles.median())

    filas = []
    for metodo, pred in [("Media global", pred_media),
                         ("Mediana global", pred_mediana),
                         ("Mediana por aerolínea", pred_grupo)]:
        mae = float(np.mean(np.abs(verdad.to_numpy() - np.asarray(pred, dtype=float))))
        filas.append({"metodo": metodo, "MAE_minutos": round(mae, 2), "valores_escondidos": n_ocultos})

    return pd.DataFrame(filas).sort_values("MAE_minutos").reset_index(drop=True)  # el mejor arriba
