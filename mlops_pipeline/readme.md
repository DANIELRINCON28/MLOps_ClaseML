# Proyecto Final - Despliegue MLOps

Este repositorio contiene el proyecto final para la clase de Machine Learning, enfocado en la implementación de un pipeline de MLOps.

## Descripción

El objetivo de este proyecto es demostrar los conceptos y prácticas de MLOps, abarcando desde la ingesta de datos hasta el monitoreo del modelo en producción.

## Estructura de Carpetas

A continuación se describe la estructura del proyecto:

```
.
PROYECTO_ML/
├── mlops_pipeline/
    ├── .gitignore              
    ├── Base_de_datos.csv       # Conjunto de datos inicial.
    ├── config.json             # Archivo de configuración.
    ├── readme.md               # README específico de la carpeta mlops_pipeline.
    ├── requeriments.txt        # Dependencias de Python.
    ├── set_up.bat              # Script de configuración inicial.
    └── src/                    # Código fuente del pipeline.
        ├── Cargar_datos.ipynb          # Notebook para la carga de datos.
        ├── Comprension_eda.ipynb       # Notebook para Análisis Exploratorio de Datos (EDA).
        ├── ft_engineering.py           # Script para ingeniería de características.
        ├── heuristic_model.py          # Script para un modelo heurístico base.
        ├── model_deploy.ipynb          # Notebook para el despliegue del modelo.
        ├── model_evaluation.ipynb      # Notebook para la evaluación del modelo.
        ├── model_monitoring.ipynb      # Notebook para el monitoreo del modelo.
        └── model_training.ipynb        # Notebook para el entrenamiento del modelo.
```


