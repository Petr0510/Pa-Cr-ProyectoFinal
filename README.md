# Pa-Cr-ProyectoFinal
Proyecto predictorio IBM (1980-2025)

# IBM Stock ML App

Proyecto de predicción de precios de acciones de IBM (1980–2025) con **Machine Learning y Streamlit**.

## Funcionalidades
- Análisis exploratorio de datos (EDA) interactivo con Plotly.
- Preprocesamiento con Pipelines, imputación y PCA.
- Modelos de ML: Regresión Lineal y Random Forest.
- Despliegue web con Streamlit.

## Estructura

```
Pa-Cr-ProyectoFinal_simplified/
├─ notebooks/
│  ├─ 1_EDA.ipynb
│  └─ 2_Modeling.ipynb
├─ app/
│  └─ app.py
├─ data/
│  └─ IBM_Stock_1980_2025.csv   
├─ models/
│  └─ (se guardarán los modelos .joblib aquí)
├─ requirements.txt
└─ README.md
```

## Ejecución
1. Crear y activar entorno virtual:
```bash

MacOs/Linux

   python -m venv .venv
   source .venv/bin/activate

Windows

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

```
2. Instalar dependencias:
```bash
   pip install -r requirements.txt
```
3. Entrenamiento:
```bash
   python -m src.model_training
```
4. Ejecucion aplicación:

```bash
   streamlit run app/streamlit_app.py
```

Autor

Cristian Ávila y Pablo Troncoso — Proyecto final con buenas prácticas de ingeniería de datos.
