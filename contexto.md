# CONTEXTO DEL PROYECTO MLOPS - MACHINE LEARNING

## 1. Resumen del Proyecto

[cite_start]**Objetivo:** Desarrollar un proyecto final para la materia "Machine Learning" [cite: 2, 3] que implementa un ciclo de vida MLOps completo. [cite_start]El entregable final no es un notebook, sino un repositorio de GitHub [cite: 15] [cite_start]que contiene un sistema automatizado para entrenar, desplegar, monitorear y asegurar la calidad de un modelo de machine learning predictivo y supervisado[cite: 7].

**Componentes Clave:**
* **Modelo:** Un modelo de clasificación supervisada.
* [cite_start]**Calidad de Código:** Integración con SonarCloud para análisis estático de calidad, seguridad y cobertura[cite: 120, 121, 126].
* [cite_start]**Despliegue:** Un API (FastAPI/Flask) para servir el modelo [cite: 111][cite_start], contenido en una imagen Docker[cite: 113].
* [cite_start]**Monitoreo:** Un script para detectar *data drift* [cite: 83] [cite_start]y un dashboard en Streamlit [cite: 81] para visualizarlo.

## 2. Dataset

* **Fuente:** Kaggle - Synthetic Financial Datasets For Fraud Detection.
* [cite_start]**Archivo en Repositorio:** `Base_de_datos.csv`  (Esta es una muestra reducida del original).
* **Características Clave:**
    * **Filas:** 200,001
    * **Tamaño:** ~15 MB (optimizado desde 470 MB).
    * **Problema Central:** **Desbalanceo Extremo**.
        * No Fraude (0): 99.87%
        * Fraude (1): 0.129%
        * **Ratio:** 1:774

* **Implicación Inmediata:** El `accuracy` es una métrica inútil. El éxito debe medirse con **Precision, Recall, F1-Score** y **AUC-PR** (Area Under Precision-Recall Curve). Se deben usar técnicas como **SMOTE** o `class_weight` en el entrenamiento.

## 3. Estructura de Repositorio (Requerida)


## 4. Componentes y Flujo de Trabajo

### 4.1. Entorno (`setup.bat` y `config.json`)
* El `setup.bat` es un script de automatización que:
    1.  Lee el `"project_code"` desde `config.json`.
    2.  Crea un entorno virtual (ej: `nombre-del-proyecto-venv`).
    3.  Instala las librerías de `requirements.txt`.
    4.  Registra este `venv` como un kernel de Jupyter para asegurar consistencia entre el EDA y la producción.
* **Acción Requerida:** No modificar el `.bat`. Solo mantener `requirements.txt` actualizado.

### 4.2. Fase de Exploración (`.ipynb`)
* **`Comprension_eda.ipynb`:** El único lugar para "explorar".
* **Objetivo:** Analizar variables, entender el desbalanceo e **identificar las transformaciones** necesarias (imputación, encoding) [cite: 39-55].

### 4.3. Fase de Producción - Scripts (`.py`)
* **Desafío Principal:** Refactorizar los hallazgos de los notebooks (`.ipynb`) a scripts de Python (`.py`) modulares y robustos.

* **`ft_engineering.py`:**
    * Debe contener el pipeline de preprocesamiento (ej: `ColumnTransformer` de Scikit-learn)[cite: 60, 62].
    * Debe empaquetar toda la lógica (imputación para numéricas [cite: 66], OneHotEncoder para categóricas [cite: 69], OrdinalEncoder para ordinales [cite: 70]).

* **`model_training_evaluation.py`:**
    * Importa el pipeline de `ft_engineering.py`.
    * Entrena y evalúa *múltiples* modelos[cite: 71].
    * Debe incluir pasos para manejar el desbalanceo (ej: `imblearn.pipeline`).
    * **Salida:** Guarda el *mejor* modelo (ej: `modelo_final.pkl`) y un reporte de métricas (ej: `metrics.json`)[cite: 72].

* **`model_deploy.py`:**
    * *No* es un notebook. Es el script de la API (FastAPI recomendado)[cite: 111].
    * Carga el `modelo_final.pkl`.
    * Define un endpoint `/predict` que recibe datos y retorna predicciones.
    * Debe soportar **predicción por lotes** (batch)[cite: 112].

* **`Dockerfile`:**
    * Debe ser creado para "envolver" el `model_deploy.py` y sus dependencias[cite: 113].
    * Define la imagen que se usará para el despliegue [cite: 115-119].

### 4.4. Fase de Monitoreo (`.py` + Streamlit)
* **`model_monitoring.py`:**
    * Un script que se ejecutaría periódicamente (simulado).
    * Compara los datos de entrenamiento con datos "actuales" para detectar *data drift*.
    * Debe calcular métricas como: **PSI (Population Stability Index)** [cite: 88], **Test de Kolmogorov-Smirnov (KS)** [cite: 87] y **Chi-cuadrado**[cite: 90].

* **Streamlit Dashboard (`app.py` - nombre sugerido):**
    * Una aplicación web separada[cite: 81].
    * Lee los resultados del `model_monitoring.py`.
    * Visualiza las métricas de drift, idealmente con alertas (semáforos) si se superan umbrales [cite: 93-96, 101].

### 4.5. Calidad (`SonarCloud`)
* Configurado a través de `sonar-project.properties` y `.github/workflows/sonarcloud.yml`.
* Analizará el código en cada *push* para detectar:
    * Bugs y Vulnerabilidades[cite: 126].
    * Código duplicado[cite: 122].
    * Complejidad ciclomática[cite: 123].
    * Cobertura de pruebas[cite: 130].

## 5. Metas Inmediatas
1.  **Refactorizar:** Mover toda la lógica de `model_training.ipynb`, `model_evaluation.ipynb`, `model_deploy.ipynb` y `model_monitoring.ipynb` a los archivos `.py` requeridos en la estructura.
2.  **Pipeline de Features:** Construir el `ColumnTransformer` en `ft_engineering.py`.
3.  **Manejo del Desbalanceo:** Integrar `imblearn` (ej: SMOTE) en el pipeline de entrenamiento.