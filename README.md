# IBM Stock ML App

Proyecto de predicción de precios de acciones de IBM (1980–2025) con **Machine Learning y Streamlit**.

## Funcionalidades
- Análisis exploratorio de datos (EDA) interactivo con Plotly.
- Preprocesamiento con Pipelines, imputación y PCA.
- Modelos de ML: Regresión Lineal y Random Forest.
- Despliegue web con Streamlit.

## Estructura

```
Pa-Cr-ProyectoFinal/
├─ notebooks/
│  ├─ 1_EDA.ipynb
│  └─ 2_Modelamiento.ipynb
├─ app/
│  └─ app.py
├─ data/
│  └─ IBM_Stock_1980_2025.csv   
├─ Models/
│  └─ (se guardarán los modelos .joblib aquí)
├─ requirements.txt
└─ README.md
```
## Clonar repositorio

```bash
   git clone https://github.com/Petr0510/Pa-Cr-ProyectoFinal.git
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
3. Ejecutar Notebooks:
```bash
   - EDA.ipynb --> Analisis exporatorio del dataset "IBM_Stock_1980_2025"
   - Modelamiento.ipynb --> Al terminar, los pipelines guardados estarán en Models/:

      * Models/LinearRegression.joblib
      * Models/RandomForest.joblib
```
4. Ejecucion aplicación:

```bash
   streamlit run app/app.py
```

Autores

Cristian Ávila y Pablo Troncoso — Proyecto final con buenas prácticas de ingeniería de datos.
