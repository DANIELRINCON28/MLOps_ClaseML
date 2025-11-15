# Carpeta mlops_pipeline

Esta carpeta contiene todo el pipeline de MLOps del proyecto.

## Estructura:

- **src/**: Todo el código fuente del proyecto (scripts Python y Notebooks Jupyter)

## Contenido de src/:

### Scripts Python (Producción):
- `ft_engineering.py`: Ingeniería de características y preprocesamiento
- `model_training_evaluation.py`: Entrenamiento y evaluación de modelos
- `model_monitoring.py`: Monitoreo de modelos en producción
- `run_full_pipeline.py`: Ejecuta el pipeline completo
- `app_monitoring.py`: Aplicación de monitoreo con Streamlit

### Notebooks Jupyter (Desarrollo):
- `Cargar_datos.ipynb`: Carga inicial de datos
- `Comprension_eda.ipynb`: Análisis exploratorio
- `Comprension_eda_completo.ipynb`: EDA completo
- `model_training.ipynb`: Entrenamiento interactivo
- `model_evaluation.ipynb`: Evaluación de modelos
- `model_monitoring.ipynb`: Monitoreo interactivo
- `model_deploy.ipynb`: Deployment

## Ejecución:

```bash
# Desde la raíz del proyecto
python main.py

# O directamente
python mlops_pipeline/src/run_full_pipeline.py
```
