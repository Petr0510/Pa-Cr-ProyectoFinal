import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from src.data_preprocessing import load_data

st.set_page_config(page_title="Predicción de Precio IBM", layout="wide")

st.title("📈 Predicción de Precio Ajustado (Adj_Close) IBM 1980–2025")

df = load_data('../data/IBM_Stock_1980_2025.csv')

tab1, tab2, tab3 = st.tabs(["Exploración", "Modelos", "Predicción"])

with tab1:
    st.header("Exploración de Datos")
    fig = px.line(df, x='Date', y='Adj_Close', title='Precio Ajustado (Adj_Close)')
    st.plotly_chart(fig, use_container_width=True)
    st.write("Estadísticas básicas:")
    st.dataframe(df.describe())

with tab2:
    st.header("Modelos Entrenados")
    st.markdown("- **Regresión Lineal:** `linear_regression_adjclose.pkl`")
    st.markdown("- **Random Forest:** `random_forest_adjclose.pkl`")

with tab3:
    st.header("Predicción del Precio")
    open_val = st.number_input("Precio de Apertura", min_value=0.0, step=0.1)
    high_val = st.number_input("Precio Máximo", min_value=0.0, step=0.1)
    low_val = st.number_input("Precio Mínimo", min_value=0.0, step=0.1)
    vol_val = st.number_input("Volumen", min_value=0.0, step=1000.0)
    year = st.number_input("Año", min_value=1980, max_value=2025, step=1)
    month = st.number_input("Mes", min_value=1, max_value=12, step=1)
    day = st.number_input("Día", min_value=1, max_value=31, step=1)

    model_choice = st.selectbox("Seleccionar Modelo", ["Linear Regression", "Random Forest"])
    model_path = "../models/linear_regression_adjclose.pkl" if model_choice == "Linear Regression" else "../models/random_forest_adjclose.pkl"

    if st.button("Predecir Precio"):
        model = joblib.load(model_path)
        features = pd.DataFrame([[open_val, high_val, low_val, vol_val, year, month, day]],
                                columns=['Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day'])
        pred = model.predict(features)[0]
        st.success(f"💰 Precio estimado (Adj Close): **${pred:.2f} USD**")