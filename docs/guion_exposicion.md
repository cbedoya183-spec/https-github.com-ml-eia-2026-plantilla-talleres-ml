# Guion de la exposición (sesión 9)

En equipos de 4 el profesor escoge quién expone, así que cualquiera debe poder recorrer todo. El orden es el de la diapositiva *"La exposición, en orden"*. Tiempo sugerido: 12–15 minutos.

## Antes de entrar
- [ ] `git pull` en la rama y `pip install -r requirements.txt && pip install -e .` en el portátil con el que se expone.
- [ ] Los 12 CSV en `data/raw/` y la muestra ya creada (`python src/data/download_data.py`).
- [ ] Correr una vez `Kernel → Restart & Run All` y confirmar que las cifras coinciden con las láminas (semilla 42).
- [ ] Plan B si el tiempo es corto: `python src/models/train.py --rapido` (menos árboles) y `models/best_model.joblib` ya guardado.

## 1 · El problema y la regla a mano (≈2 min) — notebook §1
- "Queremos saber **antes del despegue** si un vuelo llegará 15 min tarde o más (estándar FAA)."
- Por qué ML: hora (efecto cascada), temporada, hubs y aerolínea se combinan de forma no lineal.
- Mostrar la celda con `regla_manual`: *sale ≥ 16:00 y toca un hub*. "Es el número que el modelo debe vencer."
- Línea base: siempre "puntual" acierta ~78 % con recall 0 → **la exactitud no sirve** para decidir.

## 2 · La página de los datos, abierta (≈2 min)
- Abrir en el navegador la página de Kaggle y la del BTS (`transtats.bts.gov`). Quién publica y licencia (dominio público).
- Cómo se descarga: API de Kaggle en `src/data/download_data.py` (un CSV por mes).
- Qué es una fila (un vuelo programado), cuántas hay (7,8 millones crudas) y cuál es la etiqueta (`ARR_DEL15`).
- Complejidad: **12 archivos crudos** auditados y unidos en una tabla propia; muestra del 1 % por mes que respeta la estacionalidad (gráfica de cobertura).

## 3 · Limpieza y columnas (≈4 min) — notebook §3 y §4
- Tabla de auditoría: cancelados 145 mil, desviados 21 mil, 3 duplicados, 4 duraciones vacías.
- **Fechas en dos formatos** mezclados → `convertir_fecha()`.
- **Imputación comparada** (como la edad del Titanic): media vs mediana vs mediana por aerolínea → gana la de menor MAE, dentro del Pipeline.
- **Fuga:** mostrar la celda con y sin `DEP_DEL15` (AUC > 0,9 vs ~0,6). "Como `decision_renovacion` en seguros."
- Tabla antes/después: qué columnas entran, cuáles se crean (seno/coseno, hubs, franja...) y cuáles se descartan.
- Diagrama del Pipeline: todo lo que aprende va adentro → cero fuga entre pliegues.

## 4 · El código corriendo (≈3 min) — notebook §5
- Partición **temporal** (no aleatoria): "entrenar con 2026 para predecir 2025 sería trampa".
- La celda del `for` que valida todos los modelos con el **mismo `TimeSeriesSplit`**: dejar que se vea la salida en vivo.
- Umbral elegido por F1 en el último pliegue (el 0,50 no es decisión de negocio).

## 5 · Métricas y modelo elegido (≈3 min) — notebook §5.4 a §6
- Gráfica de barras de AUC con desviación: los modelos quedan muy cerca.
- **Elección:** mayor AUC, pero si un modelo más simple queda a < 0,01, gana el simple (como Ridge vs XGBoost en las casas) → **regresión logística**.
- Tabla final en la prueba con línea base y regla manual: F1 ~0,41 vs 0,28 de la regla; recall ~80 %.
- Matriz de confusión y coeficientes (odds ratio): "asociación, no causa".
- Límites: sin clima ni avión entrante, el AUC tiene techo (~0,65). Siguiente paso: pronóstico NOAA y rotación de la aeronave.

## Preguntas probables
| Pregunta | Respuesta corta |
|---|---|
| ¿Por qué ROC-AUC y no exactitud? | Clases desbalanceadas: la línea base ya tiene 78 % de exactitud sin detectar nada. AUC mide el ordenamiento del riesgo. |
| ¿Por qué no K-fold aleatorio? | Metería meses futuros en el entrenamiento; `TimeSeriesSplit` siempre valida con lo que sigue. |
| ¿Por qué `class_weight="balanced"`? | Sin él, el modelo casi nunca predice "retraso" (22 % de la clase). |
| ¿Por qué no KNN / SVM / red neuronal? | Costo de predicción con 57 mil filas en one-hot; la frontera lineal ya está en la logística y la no lineal en los ensambles. |
| ¿La prueba se usó para elegir algo? | No: modelo, umbral e hiperparámetros salen solo del entrenamiento. |
| ¿Por qué 1 % y no todo? | Para correr en vivo; la muestra conserva el peso y la tasa de retraso de cada mes (tabla `muestra_vs_mes`). |
