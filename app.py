"""
Predicción de litros por vaca (L_ACT) — TP Inteligencia Artificial y Aprendizaje Automático, UNRaf 2026.

App de Streamlit que carga el modelo Gradient Boosting entrenado en el notebook
(notebooks/TP2_Regresion_LACT.ipynb) y predice los litros por día de una vaca
en el próximo control lechero.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RUTA_MODELO = Path(__file__).parent / "modelo_l_act.joblib"
MAE_PRUEBA = 2.87  # error típico del modelo en las 61 vacas de prueba (litros/día)

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
         "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

# Resultados en el conjunto de prueba (ver notebook)
MATRIZ = pd.DataFrame(
    {
        "RMSE entren.": [3.27, 3.70, 3.73, 6.44],
        "RMSE val. cruzada": [3.68, 3.79, 3.97, 6.44],
        "RMSE prueba": [3.81, 3.85, 4.08, 6.63],
        "MAE prueba": [2.87, 2.88, 3.11, 5.23],
        "MSE prueba": [14.52, 14.83, 16.67, 43.89],
        "R² prueba": [0.668, 0.661, 0.619, 0.000],
        "MAPE prueba (%)": [13.0, 13.1, 14.0, 24.2],
        "Brecha": [0.54, 0.15, 0.35, np.nan],
    },
    index=["Gradient Boosting", "Ridge polinómica", "Árbol de decisión", "Promedio (referencia)"],
)


@st.cache_resource
def cargar_modelo():
    """Carga el pipeline entrenado (imputación + escalado + Gradient Boosting) y la lista de variables."""
    paquete = joblib.load(RUTA_MODELO)
    return paquete["modelo"], paquete["variables"]


def armar_variables(del_dias, nro_parto, l_ant, gr_ant, pr_ant, ges, mes, cs_ant):
    """Construye las mismas variables derivadas que se usaron para entrenar el modelo."""
    return {
        "DEL": del_dias,
        "NRO_PARTO": nro_parto,
        "L_ANT": l_ant,
        "GR_ANT": gr_ant if gr_ant is not None else np.nan,
        "PR_ANT": pr_ant if pr_ant is not None else np.nan,
        "GES": ges,
        "DEL_log": np.log1p(del_dias),
        "DEL_exp": np.exp(-0.05 * del_dias),
        "MES_sin": np.sin(2 * np.pi * mes / 12),
        "MES_cos": np.cos(2 * np.pi * mes / 12),
        "VERANO": int(mes in (12, 1, 2, 3)),
        # Los datos que faltan (None) quedan como NaN y el pipeline los imputa con la mediana
        "CS_ANT_log": np.log1p(cs_ant) if cs_ant is not None else np.nan,
    }


# ---------------------------------------------------------------- Interfaz
st.set_page_config(page_title="Predicción de litros por vaca", page_icon="🐄", layout="centered")
st.title("🐄 Predicción de litros por vaca")
st.write(
    "Estima los **litros por día** que va a dar una vaca en el **próximo control lechero**, "
    "a partir de datos que se conocen antes del control. Modelo: Gradient Boosting entrenado "
    "con 5.993 controles de 305 vacas de un tambo de Rafaela (2020–2024)."
)

modelo, variables = cargar_modelo()

with st.form("datos_vaca"):
    st.subheader("Datos de la vaca")
    rp = st.text_input("RP de la vaca (opcional, solo para identificarla)", "")
    c1, c2 = st.columns(2)
    with c1:
        l_ant = st.number_input("Litros en el control anterior", 0.0, 70.0, 28.5, 0.1)
        del_dias = st.number_input("Días en leche (DEL)", 1, 1300, 65, 1)
        nro_parto = st.number_input("Número de parto", 1, 12, 2, 1)
        ges = st.number_input("Días de gestación (0 si está vacía)", 0, 290, 0, 1)
    with c2:
        mes_nombre = st.selectbox("Mes del control", MESES, index=1)
        gr_ant = st.number_input("% grasa del control anterior (vacío si no hay dato)", 1.0, 7.0, None, 0.01)
        pr_ant = st.number_input("% proteína del control anterior (vacío si no hay dato)", 2.0, 5.0, None, 0.01)
        cs_ant = st.number_input("Células somáticas del control anterior (miles/ml, vacío si no hay dato)", 0, 30000, None, 10)
    real = st.number_input("Litros reales del control (opcional, para comparar)", 0.0, 80.0, None, 0.1)
    enviar = st.form_submit_button("Predecir", type="primary")

if enviar:
    mes = MESES.index(mes_nombre) + 1
    fila = pd.DataFrame([armar_variables(del_dias, nro_parto, l_ant, gr_ant, pr_ant, ges, mes, cs_ant)])[variables]
    pred = float(modelo.predict(fila)[0])

    st.subheader(f"Resultado — vaca {rp}" if rp.strip() else "Resultado")
    m1, m2 = st.columns(2)
    m1.metric("Litros predichos", f"{pred:.1f} L/día", f"{pred - l_ant:+.1f} L vs. control anterior")
    m2.metric("Rango probable (± error típico)", f"{pred - MAE_PRUEBA:.1f} – {pred + MAE_PRUEBA:.1f} L")
    if real is not None:
        r1, r2 = st.columns(2)
        r1.metric("Litros reales", f"{real:.1f} L/día")
        r2.metric("Error del modelo", f"{abs(pred - real):.1f} L")
    st.caption(
        "El error típico del modelo con vacas nuevas es de ±2,9 L. Si la producción real queda muy por "
        "debajo del rango, conviene revisar a la vaca (salud de ubre, alimentación, estrés calórico)."
    )

with st.expander("Matriz de resultados de los modelos"):
    st.dataframe(MATRIZ.style.format("{:.2f}", na_rep="—"), width="stretch")
    st.caption("Errores en litros por día, medidos en 61 vacas que el modelo nunca vio. Menor es mejor, salvo R².")

with st.expander("Limitaciones"):
    st.markdown(
        "- Necesita los litros del control anterior: no sirve para el primer control de una lactancia.\n"
        "- Hay muy pocos datos de los primeros 30 días en leche; en esa etapa la predicción es menos confiable.\n"
        "- Tiende a subestimar a las vacas de más de 40 L/día."
    )
