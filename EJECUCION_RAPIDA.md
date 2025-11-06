# 🚀 Guía Rápida de Ejecución

## ✅ Ejecutar TODO el Pipeline con UN SOLO COMANDO

### ⭐ Opción 1: Python directamente (MÁS RÁPIDO Y SIMPLE)

```bash
# Activa el ambiente virtual
MLOPS_FINAL-venv\Scripts\activate

# Ejecuta TODO el pipeline completo
python run_mlops.py

# O ejecuta TODO + abre el dashboard automáticamente
python run_mlops.py --dashboard
```

**¿Qué hace este comando?**
1. ✅ Verifica si los datos están procesados (si no, ejecuta feature engineering)
2. ✅ Entrena el modelo XGBoost con SMOTE
3. ✅ Realiza monitoreo de drift en 32 variables
4. ✅ Genera predicciones y alertas
5. ✅ (Con --dashboard) Abre el dashboard de Streamlit automáticamente

### Opción 2: Usando el archivo .bat

```bash
ejecutar_mlops.bat
```

Este script activa el ambiente virtual automáticamente y ejecuta todo el proceso.

## 📋 Qué hace cada comando

| Comando | Descripción |
|---------|-------------|
| `ejecutar_mlops.bat` | Ejecuta todo el pipeline con menú interactivo |
| `python run_mlops.py` | Ejecuta todo el pipeline (solo procesamiento) |
| `python run_mlops.py --dashboard` | Ejecuta pipeline + abre dashboard |

## 📊 Archivos generados

Después de ejecutar, encontrarás:

```
├── models/
│   ├── xgboost_model.pkl          # Modelo entrenado
│   └── model_metrics.pkl           # Métricas del modelo
├── data/processed/
│   ├── X_train.pkl                 # Datos de entrenamiento
│   ├── X_test.pkl                  # Datos de prueba
│   └── preprocessor.pkl            # Preprocesador
└── outputs/monitoring/
    ├── drift_results_*.csv         # Resultados de drift
    ├── predictions_*.csv           # Predicciones
    └── alerts_*.json               # Alertas generadas
```

## ⚡ Atajos rápidos

### Solo entrenar el modelo
```bash
cd mlops_pipeline\src
python run_full_pipeline.py
```

### Solo ejecutar monitoreo
```bash
cd mlops_pipeline\src
python model_monitoring.py
```

### Solo abrir el dashboard
```bash
streamlit run app_monitoring.py
```

## 🔧 Solución de problemas

### Error: "No se encontró el ambiente virtual"
```bash
# Ejecuta primero el setup
set_up.bat
```

### Error: "Module not found"
```bash
# Activa el ambiente e instala dependencias
MLOPS_FINAL-venv\Scripts\activate
pip install -r requirements.txt
```

### El dashboard no abre
```bash
# Verifica que Streamlit esté instalado
pip install streamlit

# Ejecuta manualmente
streamlit run app_monitoring.py
```

## 📝 Notas

- El pipeline completo toma aproximadamente 2-5 minutos
- Asegúrate de tener el archivo `Base_datos.csv` en el directorio raíz
- El dashboard se abre en http://localhost:8501

---

**¿Tienes dudas?** Revisa el archivo `README_COMPLETO.md` para más detalles.
