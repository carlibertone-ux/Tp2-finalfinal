# Predicción de litros por vaca con aprendizaje automático

Trabajo práctico de **Inteligencia Artificial y Aprendizaje Automático** — UNRaf, 2026.

Modelo de regresión que predice los **litros por día de cada vaca en el próximo control lechero** (`L_ACT`) de un tambo de Rafaela, usando solo datos conocidos antes del control.

**App online:** _(pegar acá el link de Streamlit)_

## Datos

- 5.993 controles lecheros de 305 vacas (2020–2024).
- Variables usadas: litros del control anterior, días en leche, número de parto, días de gestación, mes del control, y % de grasa, % de proteína y células somáticas del control anterior.
- Partición **por vaca**: 80 % para entrenamiento (244 vacas) y 20 % para prueba (61 vacas), más validación cruzada GroupKFold (k = 5).
- El dataset pertenece al tambo y no se publica en este repositorio.

## Modelos comparados

| Modelo | RMSE val. cruzada | RMSE prueba | MAE prueba | R² prueba | Brecha |
| --- | --- | --- | --- | --- | --- |
| **Gradient Boosting** | **3,68** | **3,81** | **2,87** | **0,668** | 0,54 |
| Ridge polinómica | 3,79 | 3,85 | 2,88 | 0,661 | **0,15** |
| Árbol de decisión | 3,97 | 4,08 | 3,11 | 0,619 | 0,35 |
| Promedio (referencia) | 6,44 | 6,63 | 5,23 | 0,000 | — |

Errores en litros por día. El modelo elegido es **Gradient Boosting**: se equivoca en promedio ±2,9 L por vaca, un 45 % menos que predecir siempre el promedio del rodeo.

## Estructura

```
├── app.py                         # App de Streamlit
├── modelo_l_act.joblib            # Modelo entrenado (pipeline completo)
├── requirements.txt               # Dependencias
└── notebooks/
    └── TP2_Regresion_LACT.ipynb   # Preparación, entrenamiento y evaluación
```

## Cómo correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para reentrenar, poner `Dataset_Ordenado_Por_Fecha.xlsx` junto al notebook y ejecutar todas las celdas en orden. La última celda vuelve a generar `modelo_l_act.joblib`. Si se reentrena, la versión de `scikit-learn` en `requirements.txt` tiene que coincidir con la usada al guardar el modelo.

## Limitaciones

- Necesita los litros del control anterior: no predice el primer control de una lactancia.
- Hay pocos datos de los primeros 30 días en leche.
- Tiende a subestimar a las vacas de más de 40 L/día.
