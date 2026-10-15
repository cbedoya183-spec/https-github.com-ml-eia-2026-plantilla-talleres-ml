"""
pipeline.py — PASO 4/5: preprocesamiento + modelo dentro de UN solo Pipeline.

¿Por qué todo dentro del Pipeline? Porque cuando se llama pipe.fit(X_train),
cada paso aprende (medianas, medias, categorías) SOLO con X_train. Luego
pipe.predict(X_test) reutiliza lo aprendido sin volver a mirar la prueba.
Así ningún dato de prueba se filtra al entrenamiento (cero fuga).

Esquema:
    Pipeline
     ├── 1. imputar_duracion  -> GroupMedianImputer (mediana por aerolínea)
     ├── 2. preprocesamiento  -> ColumnTransformer
     │        ├── numéricas:   mediana  -> StandardScaler
     │        └── categóricas: "DESCONOCIDO" -> OneHotEncoder (top 25 + "infrequent")
     └── 3. modelo            -> cualquier clasificador de scikit-learn / XGBoost
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS, TOP_AEROPUERTOS
from src.features.transformers import GroupMedianImputer


def crear_preprocesador() -> ColumnTransformer:
    """Arma el ColumnTransformer: un tratamiento para números y otro para categorías."""
    numericas = Pipeline(steps=[
        ("imputar", SimpleImputer(strategy="median")),   # vacíos -> mediana del entrenamiento
        ("escalar", StandardScaler()),                   # (x - media) / desviación: misma escala
    ])                                                   # (la logística lo necesita; los árboles no)

    categoricas = Pipeline(steps=[
        ("imputar", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
        ("one_hot", OneHotEncoder(
            max_categories=TOP_AEROPUERTOS + 1,          # conserva las 25 más frecuentes ...
            handle_unknown="infrequent_if_exist",        # ... y el resto (o una nueva) va a "infrequent"
            sparse_output=False,                         # matriz densa (la aceptan todos los modelos)
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numericas, COLUMNAS_NUMERICAS),     # se aplica a las columnas numéricas
            ("cat", categoricas, COLUMNAS_CATEGORICAS), # se aplica a las columnas categóricas
        ],
        remainder="drop",                               # cualquier otra columna se ignora
        verbose_feature_names_out=False,                # nombres limpios: "ORIGIN_ATL", no "cat__ORIGIN_ATL"
    )


def crear_pipeline(modelo) -> Pipeline:
    """Une los tres pasos. El mismo preprocesamiento sirve para todos los modelos."""
    return Pipeline(steps=[
        ("imputar_duracion", GroupMedianImputer()),     # paso 1: duración por aerolínea
        ("preprocesamiento", crear_preprocesador()),    # paso 2: números y categorías
        ("modelo", modelo),                             # paso 3: el clasificador
    ])
