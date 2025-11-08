# 🚀 Guía de Instalación Rápida - MLOps Detección de Fraude

## 📋 Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.8 o superior** - [Descargar aquí](https://www.python.org/downloads/)
- **Git** (opcional, solo si clonas el repositorio)

## 🔧 Instalación en un Nuevo PC

### Paso 1: Descargar el Proyecto

Clona el repositorio o descarga el ZIP:

```bash
git clone https://github.com/DANIELRINCON28/MLOps_ClaseML.git
cd MLOps_ClaseML
```

O descarga y extrae el archivo ZIP.

### Paso 2: Configurar el Ambiente

Ejecuta el script de configuración (SOLO LA PRIMERA VEZ):

```bash
set_up.bat
```

Este script automáticamente:
- ✅ Crea el ambiente virtual de Python
- ✅ Instala todas las dependencias necesarias
- ✅ Registra el kernel de Jupyter
- ✅ Crea los directorios necesarios
- ✅ Verifica que todo esté correctamente configurado

**⏱️ Tiempo estimado:** 5-10 minutos (dependiendo de tu conexión a internet)

### Paso 3: Ejecutar el Proyecto

Una vez configurado el ambiente, ejecuta:

```bash
ejecutar_mlops.bat
```

Este comando ejecutará todo el pipeline de MLOps:
1. **Feature Engineering** - Procesa y prepara los datos
2. **Entrenamiento de Modelos** - Entrena 5 modelos diferentes
3. **Evaluación** - Compara y selecciona el mejor modelo
4. **Monitoreo** - Analiza drift en los datos

**⏱️ Tiempo estimado:** 10-15 minutos

Al finalizar, el **dashboard interactivo se abrirá automáticamente** en tu navegador (`http://localhost:8501`).

## 📊 Ver Resultados

El dashboard de Streamlit se abre automáticamente al finalizar el pipeline y muestra:

- 📈 **Métricas de rendimiento** de todos los modelos
- 🎯 **Comparación de modelos** entrenados
- 📊 **Análisis de drift** en los datos
- ⚠️ **Alertas y anomalías** detectadas
- 📉 **Visualizaciones interactivas** de resultados

### Para cerrar el dashboard:
- Presiona `Ctrl+C` en la terminal

### Para abrir el dashboard sin ejecutar el pipeline:
```bash
streamlit run app_monitoring.py
```

## 📁 Estructura de Resultados

Después de ejecutar el pipeline, encontrarás:

```
models/
├── xgboost_model.pkl              # Mejor modelo entrenado
├── best_model_metadata.json       # Metadata del modelo
└── model_metrics.pkl              # Métricas del modelo

outputs/
├── model_comparison.csv           # Comparación de todos los modelos
├── all_models_results.json        # Resultados detallados
└── monitoring/
    ├── predictions.csv            # Predicciones del modelo
    ├── drift_results_*.csv        # Análisis de drift
    └── alerts_*.json              # Alertas generadas
```

## 🔄 Uso Posterior

### En el mismo PC:

Ya NO necesitas ejecutar `set_up.bat` nuevamente. Solo ejecuta:

```bash
ejecutar_mlops.bat
```

### En un PC nuevo:

Simplemente repite los pasos 1-3:
1. Clonar/descargar el proyecto
2. Ejecutar `set_up.bat` (solo la primera vez)
3. Ejecutar `ejecutar_mlops.bat`

## 🛠️ Comandos Útiles

### Activar ambiente virtual manualmente:
```bash
MLOPS_FINAL-venv\Scripts\activate
```

### Ejecutar pipeline con Python:
```bash
python run_mlops.py
```

### Ver dashboard sin ejecutar pipeline:
```bash
streamlit run app_monitoring.py
```

### Ejecutar solo el monitoreo:
```bash
python mlops_pipeline\src\model_monitoring.py
```

## ❓ Solución de Problemas

### Error: "Python no está instalado"
- Instala Python 3.8+ desde [python.org](https://www.python.org/)
- Asegúrate de marcar "Add Python to PATH" durante la instalación

### Error: "No se encontró el ambiente virtual"
- Ejecuta `set_up.bat` primero
- Verifica que se creó la carpeta `MLOPS_FINAL-venv`

### Error durante instalación de paquetes:
- Verifica tu conexión a internet
- Ejecuta `set_up.bat` nuevamente y selecciona "S" para recrear el ambiente

### El dashboard no se abre:
- Verifica que Streamlit se instaló correctamente
- Ejecuta manualmente: `streamlit run app_monitoring.py`

## 📞 Soporte

Para más información, consulta:
- `README.md` - Documentación completa del proyecto
- `docs/` - Documentación detallada adicional
- Issues en GitHub: [Reportar problema](https://github.com/DANIELRINCON28/MLOps_ClaseML/issues)

## ✨ Características del Proyecto

- ✅ Pipeline MLOps completo automatizado
- ✅ Entrenamiento de múltiples modelos (5 algoritmos)
- ✅ Selección automática del mejor modelo
- ✅ Monitoreo de drift en datos
- ✅ Dashboard interactivo con Streamlit
- ✅ Detección automática de anomalías
- ✅ Generación de reportes y alertas

---

**¡Listo!** 🎉 Tu proyecto MLOps está configurado y listo para usar.
