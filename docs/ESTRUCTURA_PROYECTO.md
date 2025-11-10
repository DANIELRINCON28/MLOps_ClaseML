# 📁 Estructura del Proyecto MLOps

## 🎯 Archivos en la Raíz (Esenciales)

```
PROYECTO_ML/
├── README.md                    # Documentación principal del proyecto
├── LEER_RUBRICA.md             # Mapeo de requisitos de la rúbrica
├── main.py                     # 🚀 PUNTO DE ENTRADA PRINCIPAL
├── setup.py                    # Configuración del paquete Python
├── requirements.txt            # Dependencias del proyecto
└── Base_datos.csv             # Dataset original (PaySim)
```

## 📂 Carpetas Principales

### 1️⃣ **mlops_pipeline/** - Pipeline de MLOps
```
mlops_pipeline/
├── README.md                   # Documentación del pipeline
└── src/                        # 🐍 Todo el código (Python + Notebooks)
    ├── ft_engineering.py       # Ingeniería de características
    ├── model_training_evaluation.py  # Entrenamiento y evaluación
    ├── model_monitoring.py     # Monitoreo de modelos
    ├── run_full_pipeline.py    # Pipeline completo
    ├── run_full_pipeline_simple.py
    ├── train_multiple_models.py
    ├── app_monitoring.py       # App Streamlit de monitoreo
    ├── ejecutar_proyecto.py
    ├── run_mlops.py
    ├── README.md               # Documentación de src/
    │
    ├── Cargar_datos.ipynb      # 📓 Notebooks Jupyter
    ├── Comprension_eda.ipynb
    ├── Comprension_eda_completo.ipynb
    ├── model_training.ipynb
    ├── model_evaluation.ipynb
    ├── model_monitoring.ipynb
    └── model_deploy.ipynb
```

### 2️⃣ **config/** - Configuraciones
```
config/
├── README.md                   # Documentación de configuraciones
├── config.json                 # Configuración general del proyecto
├── pytest.ini                  # Configuración de tests
├── sonar-project.properties    # SonarCloud (calidad de código)
├── Dockerfile                  # Contenedorización
└── docker-compose.yml          # Orquestación de servicios
```

### 3️⃣ **scripts/** - Scripts de Ejecución
```
scripts/
├── README.md                   # Documentación de scripts
├── run_all.ps1                # Ejecutar pipeline (PowerShell)
├── run_all.sh                 # Ejecutar pipeline (Bash)
├── ejecutar_mlops.bat         # Batch para Windows
├── set_up.bat                 # Configuración inicial
├── check_environment.py       # Verificar entorno Python
└── check_data.py              # Validar datos
```

### 4️⃣ **tests/** - Tests Unitarios
```
tests/
├── __init__.py
├── test_feature_engineering.py  # Tests de ingeniería de características
├── test_model_training.py       # Tests de entrenamiento
├── test_monitoring.py           # Tests de monitoreo
└── test_utils.py                # Tests de utilidades
```

### 5️⃣ **models/** - Modelos Entrenados
```
models/
└── best_model_metadata.json    # Metadata del mejor modelo
```

### 6️⃣ **outputs/** - Resultados
```
outputs/
├── all_models_results.json     # Resultados de todos los modelos
├── model_comparison.csv        # Comparación de modelos
└── monitoring/                 # Resultados de monitoreo
    ├── predictions.csv
    ├── drift_results_*.csv
    ├── alerts_*.json
    └── latest_summary.json
```

### 7️⃣ **data/** - Datos Procesados
```
data/
└── processed/                  # Datos procesados
    ├── X_train.pkl
    ├── X_test.pkl
    ├── y_train.pkl
    ├── y_test.pkl
    ├── preprocessor.pkl
    ├── df_features_complete.pkl
    ├── feature_engineering_metadata.pkl
    └── temp_production_data.csv
```

### 8️⃣ **docs/** - Documentación
```
docs/
├── INDEX.md                    # Índice de documentación
├── RESUMEN_EJECUTIVO.md
├── contexto.md
├── EJECUCION_RAPIDA.md
├── QUICK_START_MONITORING.md
├── INSIGHTS.md
└── ...otros documentos técnicos
```

### 9️⃣ **api/** - API REST (Opcional)
```
api/
├── main.py                     # FastAPI endpoint
├── requirements.txt
├── test_api.py
└── README.md
```

### 🔟 **.github/** - CI/CD
```
.github/
└── workflows/
    ├── test.yml                # Tests automáticos
    └── sonarcloud.yml          # Análisis de código
```

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Desde la raíz (RECOMENDADO)
```bash
python main.py
```

### Opción 2: Usando scripts
```powershell
# Windows PowerShell
.\scripts\run_all.ps1

# Windows CMD
.\scripts\ejecutar_mlops.bat

# Linux/Mac
bash scripts/run_all.sh
```

### Opción 3: Pipeline específico
```bash
python mlops_pipeline/src/run_full_pipeline.py
```

## 📊 Ejecutar Tests
```bash
pytest tests/ -v --cov=mlops_pipeline/src -c config/pytest.ini
```

## 🔍 Análisis de Calidad
```bash
# SonarCloud (automático en GitHub Actions)
# Ver config/sonar-project.properties
```

## 🐳 Docker
```bash
# Build
docker build -f config/Dockerfile -t mlops-fraud-detection .

# Run
docker-compose -f config/docker-compose.yml up
```

## 📝 Notas Importantes

1. **main.py** es el punto de entrada principal del proyecto
2. Todo el código (Python + Notebooks) está en `mlops_pipeline/src/`
3. El **código productivo** son los archivos `.py`
4. Los **notebooks** (`.ipynb`) son para desarrollo y análisis
5. Todas las **configuraciones** están centralizadas en `config/`
6. Los **scripts de ejecución** están en `scripts/`
7. **39 tests unitarios** cubren todo el código (>80% coverage)

## 🎯 Ventajas de esta Estructura

✅ **Claridad**: Cada carpeta tiene un propósito específico
✅ **Escalabilidad**: Fácil agregar nuevos componentes
✅ **Mantenibilidad**: Código organizado y documentado
✅ **CI/CD**: Integración automática con GitHub Actions
✅ **Estándares**: Sigue mejores prácticas de MLOps
✅ **Modularidad**: Componentes independientes y reutilizables
