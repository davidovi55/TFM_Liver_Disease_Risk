# Plataforma para el análisis predictivo del riesgo de enfermedad hepática a diez años mediante aprendizaje automático

Trabajo de Fin de Máster orientado al desarrollo de una plataforma para el análisis del riesgo de aparición de enfermedad hepática a diez años mediante técnicas de minería de datos y aprendizaje automático.

El proyecto utiliza datos longitudinales de **KLoSA (Korean Longitudinal Study of Aging)** para construir una variable objetivo de incidencia de enfermedad hepática y desarrollar un pipeline completo de preparación de datos, análisis exploratorio, selección de variables, entrenamiento, evaluación e interpretación de modelos.

---

## Objetivo

El objetivo principal es estudiar la viabilidad de predecir el riesgo de desarrollar enfermedad hepática a diez años utilizando información disponible al inicio del seguimiento.

El proyecto incluye:

- Construcción de un dataset longitudinal a partir de varias olas de KLoSA.
- Definición de la variable objetivo de incidencia de enfermedad hepática.
- Análisis de calidad y limpieza de datos.
- Tratamiento de códigos especiales y valores ausentes.
- Revisión de valores imposibles y outliers.
- Clasificación de variables continuas y categóricas.
- Análisis de correlación y dimensionalidad.
- One-Hot Encoding de variables categóricas para modelos que lo requieren.
- Normalización de variables continuas.
- Comparación de diferentes algoritmos de clasificación.
- Tratamiento del fuerte desbalance entre clases.
- Optimización de hiperparámetros.
- Selección del umbral de clasificación mediante predicciones out-of-fold.
- Evaluación mediante validación cruzada y conjunto holdout.
- Preparación del modelo para su integración en una aplicación web.

---

## Pipeline del proyecto

El flujo general seguido es:

```text
Datos originales KLoSA
        ↓
Construcción longitudinal del dataset
        ↓
Definición de enfermedad hepática incidente
        ↓
Tratamiento de códigos especiales (-8 / -9)
        ↓
Limpieza y validación de variables
        ↓
Análisis de valores ausentes y outliers
        ↓
Clasificación de variables
        ↓
Análisis de dimensionalidad
        ↓
Preprocesamiento
        ↓
Entrenamiento y comparación de modelos
        ↓
Optimización mediante validación cruzada
        ↓
Selección del threshold
        ↓
Evaluación final
        ↓
Aplicación web
```

---

## Estructura del repositorio

```text
TFM_Liver_Disease_Risk/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   ├── config.py
│   ├── variables.py
│   └── variable_groups_v2.json
│
├── data/
│   ├── raw/
│   │   └── klosa/
│   │
│   ├── interim/
│   │   ├── klosa_liver_history_w1_w6.csv
│   │   ├── klosa_liver_stage1_special_codes_cleaned.csv
│   │   └── klosa_variable_availability_w1_w6.csv
│   │
│   └── processed/
│       ├── klosa_liver_incident_10y_master.csv
│       └── klosa_liver_modeling_dataset_v2.csv
│
├── models/
│   └── final_liver_risk_model_v3.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_dataset_construction.ipynb
│   ├── 03_data_quality_and_eda.ipynb
│   ├── 04_data_preparation_v2.ipynb
│   ├── 05_dimensionality_and_feature_selection.ipynb
│   └── 06_model_training_v3.ipynb
│
├── reports/
│   ├── data_quality/
│   ├── feature_selection/
│   ├── figures/
│   ├── preprocessing_v2/
│   ├── model_training_v3/
│   └── tables/
│
├── src/
│   ├── calibration.py
│   ├── dataset_builder.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── __init__.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Notebooks

### 01 — Data Understanding

Exploración inicial de los ficheros KLoSA y análisis de la disponibilidad de variables entre las distintas olas.

### 02 — Dataset Construction

Construcción del dataset longitudinal y definición de la variable objetivo de incidencia de enfermedad hepática.

### 03 — Data Quality and EDA

Auditoría de calidad de los datos:

- Valores ausentes.
- Códigos especiales de KLoSA.
- Variables constantes y cuasi-constantes.
- Potencial fuga de información.
- Distribución de la variable objetivo.

Los códigos especiales `-8` y `-9` se convierten en valores ausentes cuando representan ausencia de información.

### 04 — Data Preparation V2

Preparación del dataset definitivo utilizado para modelado.

Incluye:

- Eliminación de variables completamente vacías o constantes.
- Eliminación de variables con más del 50 % de valores ausentes.
- Validación de rangos lógicos.
- Revisión de variables de edad y porcentajes.
- Clasificación de variables continuas y categóricas.
- Auditoría de outliers mediante IQR.

Los valores extremos plausibles se conservan. Una variable no se elimina únicamente por presentar una elevada proporción de outliers estadísticos.

### 05 — Dimensionality and Feature Selection

Análisis de dimensionalidad y relaciones entre variables:

- Información mutua.
- Correlación con la variable objetivo.
- Matriz de correlación.
- Identificación de variables altamente correlacionadas.
- PCA.
- Evaluación de alternativas de selección de variables.

Aunque se estudia la reducción dimensional, el entrenamiento final también considera el conjunto completo de variables válidas para evitar perder interacciones útiles.

### 06 — Model Training V3

Entrenamiento y evaluación de modelos.

Se comparan principalmente:

- XGBoost.
- CatBoost.
- Balanced Random Forest.

XGBoost utiliza:

```text
Variables continuas
    ↓
Imputación por mediana
    ↓
StandardScaler

Variables categóricas
    ↓
Imputación por moda
    ↓
One-Hot Encoding
```

CatBoost utiliza las variables categóricas de forma nativa.

La evaluación incluye:

- Validación cruzada estratificada repetida.
- PR-AUC.
- ROC-AUC.
- Sensibilidad.
- Especificidad.
- Precisión.
- F1.
- F2.
- Balanced Accuracy.
- Matthews Correlation Coefficient.
- Curva Precision-Recall.
- Curva ROC.
- Matriz de confusión.

---

## Modelo final

Tras la comparación de los diferentes enfoques, **XGBoost** mostró el mejor comportamiento global y fue seleccionado como modelo principal.

El modelo final utiliza las variables válidas disponibles tras el proceso de limpieza y aplica el preprocesamiento dentro del pipeline para evitar fuga de información.

Debido al fuerte desbalance de clases, la selección del modelo se realiza principalmente mediante **PR-AUC**, complementada con ROC-AUC y métricas dependientes del threshold.

---

## Resultados

Los resultados muestran una capacidad discriminativa moderada.

### Validación out-of-fold

```text
PR-AUC:  0.543
ROC-AUC: 0.943
```

### Holdout

```text
PR-AUC:  0.56
ROC-AUC: 0.892
```

La diferencia entre PR-AUC y ROC-AUC está condicionada por el fuerte desbalance de la variable objetivo.

Por este motivo, el modelo no se plantea como una herramienta diagnóstica, sino como un sistema exploratorio para la **estratificación del riesgo**.

El umbral de clasificación se estudia de forma independiente utilizando predicciones out-of-fold, evaluando el compromiso entre:

- Precision.
- Recall.
- Specificity.
- F1.
- F2.
- MCC.

---

## Interpretación de los resultados

El rendimiento obtenido indica que existe cierta señal predictiva en las variables basales, aunque esta señal no es suficiente para realizar una clasificación clínica fiable de forma aislada.

La elevada dificultad del problema está relacionada con diferentes factores:

- Fuerte desbalance entre las clases.
- Número reducido de eventos positivos.
- Horizonte temporal amplio de diez años.
- Dataset no diseñado específicamente para la predicción de enfermedad hepática.
- Variables procedentes de cuestionarios y datos autodeclarados.
- Población de estudio específica.
- Ausencia de determinadas variables clínicas y analíticas directamente relacionadas con la función hepática.

El sistema debe interpretarse como una herramienta de análisis y estratificación de riesgo dentro de un contexto experimental.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd TFM_Liver_Disease_Risk
```

### 2. Crear el entorno con Conda

```bash
conda env create -f environment.yml
```

Activar el entorno:

```bash
conda activate tfm
```

También puede instalarse el entorno mediante:

```bash
pip install -r requirements.txt
```

---

## Ejecución de los notebooks

Los notebooks deben ejecutarse en el siguiente orden:

```text
01_data_understanding.ipynb
        ↓
02_dataset_construction.ipynb
        ↓
03_data_quality_and_eda.ipynb
        ↓
04_data_preparation_v2.ipynb
        ↓
05_dimensionality_and_feature_selection.ipynb
        ↓
06_model_training_v3.ipynb
```

Cada notebook genera los datasets, informes o configuraciones necesarios para las siguientes etapas.

---

## Aplicación web

El proyecto incluye una aplicación desarrollada con **Streamlit**.

Para ejecutarla:

```bash
streamlit run app/streamlit_app.py
```

La aplicación está orientada a presentar de forma comprensible el análisis de riesgo y los factores asociados.

La salida debe interpretarse como un **indicador de riesgo**, no como un diagnóstico médico.

---

## Reproducibilidad

Se utilizan semillas aleatorias fijas para garantizar la reproducibilidad de:

- División de datos.
- Validación cruzada.
- Entrenamiento de modelos.
- Búsqueda de hiperparámetros.

El preprocesamiento se ajusta exclusivamente sobre los datos de entrenamiento dentro de cada proceso de validación.

Esto evita utilizar información del conjunto de validación o evaluación durante el entrenamiento.

---

## Principales tecnologías

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Imbalanced-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

---

## Limitaciones

Las principales limitaciones del proyecto son:

1. Fuerte desequilibrio entre clases.
2. Número reducido de casos incidentes.
3. Dataset no específico de enfermedad hepática.
4. Horizonte predictivo de diez años.
5. Información parcialmente autodeclarada.
6. Población de estudio surcoreana.
7. Ausencia de algunas variables clínicas específicas de función hepática.
8. Rendimiento predictivo moderado del modelo final.

---

## Líneas futuras

Como continuación del proyecto se plantean:

- Incorporar datasets específicamente diseñados para enfermedad hepática.
- Aumentar el número de casos positivos.
- Incorporar biomarcadores hepáticos y variables analíticas.
- Realizar validación externa sobre otras poblaciones.
- Analizar diferentes horizontes temporales.
- Incorporar calibración de probabilidades.
- Profundizar en la interpretación mediante SHAP.
- Evaluar modelos temporales y longitudinales.
- Mejorar la aplicación web y la presentación personalizada de resultados.

---

## Aviso

Este proyecto ha sido desarrollado con fines académicos y de investigación.

Los resultados generados por el modelo **no constituyen un diagnóstico médico ni sustituyen la evaluación realizada por profesionales sanitarios**.

---

## Autor

**David**
 
Máster en Big Data, Data Science e Inteligencia Artificial
