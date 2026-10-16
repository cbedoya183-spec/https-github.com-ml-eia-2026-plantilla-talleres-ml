"""
test_humo.py — Prueba de humo: ¿el código corre de principio a fin?

Usa datos simulados (no los reales) en una carpeta temporal, así que no
necesita Kaggle. Correr con:   python -m pytest -q
"""

import pandas as pd

from src.data.download_data import auditar_mes, leer_mes, muestrear_mes
from src.features.build_features import preparar_tabla
from src.models.evaluate import metricas
from src.models.experiment import (
    definir_modelos,
    elegir_campeon,
    comparar_modelos,
    regla_manual,
    separar_temporal,
)
from src.models.pipeline import crear_pipeline
from tests.generar_datos_simulados import generar


def test_de_punta_a_punta(tmp_path):
    generar(tmp_path, filas=3000)                                   # 12 CSV pequeños inventados
    muestras = []
    for archivo in sorted(tmp_path.glob("*.csv")):
        df = leer_mes(archivo)
        fila = auditar_mes(df, archivo.name)
        assert fila["duplicados_clave_vuelo"] >= 2                  # el generador mete 2 duplicados
        muestras.append(muestrear_mes(df, 0.5))
    tabla = preparar_tabla(pd.concat(muestras, ignore_index=True))

    # Ninguna columna de fuga puede sobrevivir a la limpieza.
    assert "DEP_DELAY" not in tabla.columns and "ARR_DELAY" not in tabla.columns

    X_train, X_test, y_train, y_test = separar_temporal(tabla)
    assert X_train.index.max() < X_test.index.min()                 # prueba = futuro

    modelos = definir_modelos(y_train, rapido=True)
    modelos = {k: modelos[k] for k in ["Línea base", "Regresión logística", "Árbol de decisión", "Red neuronal (MLP)"]}
    tabla_cv = comparar_modelos(modelos, X_train, y_train)
    campeon, _ = elegir_campeon(tabla_cv)

    pipe = crear_pipeline(modelos[campeon]).fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    resultado = metricas(y_test, (prob >= 0.5).astype(int), prob)
    assert 0.5 < resultado["roc_auc"] <= 1.0                        # el modelo aprende algo
    assert len(regla_manual(X_test)) == len(X_test)
