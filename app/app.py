import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# Configuración inicial
st.set_page_config(page_title="Predicción de Precio IBM", layout="wide")
st.title("📈 Predicción de Precio Ajustado (Adj Close) — IBM 1980–2025")

# --- Función para cargar datos ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    # Limpieza básica
    if 'Volume' in df.columns:
        # Normalizar Volume: eliminar comas, espacios y cualquier carácter no numérico
        df['Volume'] = df['Volume'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df.loc[df['Volume'] == '', 'Volume'] = pd.NA
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Volume'] = df['Volume'].fillna(df['Volume'].median())
    # Features temporales
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df

# --- Cargar datos ---
data_path = Path("data/IBM_Stock_1980_2025.csv")
if data_path.exists():
    df = load_data(data_path)
else:
    st.error("⚠️ No se encontró el archivo 'data/IBM_Stock_1980_2025.csv'. Colócalo en la carpeta /data y recarga la app.")
    st.stop()

# --- Tabs principales ---
tab1, tab2, tab3 = st.tabs(["📊 Exploración", "🧠 Modelos", "🔮 Predicción"])

# --- TAB 1: Exploración ---
with tab1:
    st.header("Exploración de Datos")
    fig = px.line(df, x='Date', y='Close', title='Evolución del Precio de Cierre (Close)', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Estadísticas básicas")
    st.dataframe(df.describe())

    st.subheader("🔍 Correlaciones")
    corr = df.select_dtypes('number').corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Matriz de Correlación", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 2: Modelos ---
with tab2:
    st.header("Modelos Entrenados Disponibles")

    models_dir = Path("models")
    if not models_dir.exists():
        st.warning("⚠️ Aún no existe la carpeta 'models'. Ejecuta el notebook 2_Modeling.ipynb para crear los modelos.")
    else:
        models = [f.name for f in models_dir.glob("*.joblib")]
        if models:
            st.success("✅ Modelos encontrados:")
            for m in models:
                st.markdown(f"- {m}")
        else:
            st.warning("⚠️ No se han guardado modelos todavía. Entrena el proyecto para generarlos.")

    st.markdown("""
    **Modelos esperados:**
    - `LinearRegression.joblib`
    - `RandomForest.joblib`
    """)

# --- TAB 3: Predicción ---
with tab3:
    st.header("Predicción del Precio de Cierre (Close)")

    st.markdown("Ingresa los valores para generar una predicción:")

    col1, col2, col3 = st.columns(3)
    with col1:
        open_val = st.number_input("Precio de Apertura (Open)", min_value=0.0, step=0.1)
        high_val = st.number_input("Precio Máximo (High)", min_value=0.0, step=0.1)
    with col2:
        low_val = st.number_input("Precio Mínimo (Low)", min_value=0.0, step=0.1)
        vol_val = st.number_input("Volumen (Volume)", min_value=0.0, step=1000.0)
    with col3:
        year = st.number_input("Año", min_value=1980, max_value=2025, step=1)
        month = st.number_input("Mes", min_value=1, max_value=12, step=1)
        day = st.number_input("Día", min_value=1, max_value=31, step=1)

    model_choice = st.selectbox("Seleccionar Modelo", ["LinearRegression", "RandomForest"], index=1)
    model_path = Path(f"models/{model_choice}.joblib")

    if st.button("Predecir Precio"):
        if not model_path.exists():
            st.error(f"⚠️ No se encontró el modelo {model_choice}. Entrénalo primero en el notebook 2_Modeling.ipynb.")
        else:
            model = joblib.load(model_path)
            features = pd.DataFrame([[open_val, high_val, low_val, vol_val, year, month, day]],
                                    columns=['Open', 'High', 'Low', 'Volume', 'year', 'month', 'day'])
            pred = model.predict(features)[0]
            st.success(f"💰 Precio estimado (Close): **${pred:.2f} USD**")

            st.markdown("---")
            st.info("📌 Consejo: prueba con diferentes valores para ver cómo cambian las predicciones.")

