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
2. ✅ Entrena **5 MODELOS DIFERENTES**:
   - Logistic Regression
   - Random Forest
   - XGBoost  
   - LightGBM
   - Gradient Boosting
3. ✅ Selecciona automáticamente el mejor modelo basado en ROC-AUC
4. ✅ Realiza monitoreo de drift en 32 variables
5. ✅ Genera predicciones y alertas
6. ✅ (Con --dashboard) Abre el dashboard de Streamlit automáticamente

### Opción 2: Usando el archivo .bat

```bash
ejecutar_mlops.bat
```

Este script activa el ambiente virtual automáticamente y ejecuta todo el proceso.

## 📋 Qué hace cada comando

| Comando | Descripción | Modelos | Tiempo aprox. |
|---------|-------------|---------|---------------|
| `python run_mlops.py` | Pipeline completo (sin dashboard) | 5 modelos | ~5-8 minutos |
| `python run_mlops.py --dashboard` | Pipeline + Dashboard | 5 modelos | ~5-8 min + dashboard |
| `ejecutar_mlops.bat` | Pipeline con menú interactivo | 5 modelos | ~5-8 minutos |

### 🤖 Modelos Entrenados

El pipeline entrena y compara automáticamente 5 modelos de Machine Learning:

1. **Logistic Regression** - Modelo lineal base, rápido y simple
2. **Random Forest** - Ensemble de 100 árboles de decisión
3. **XGBoost** - Gradient Boosting optimizado (200 estimadores)
4. **LightGBM** - Gradient Boosting ligero y eficiente (150 estimadores)
5. **Gradient Boosting** - Gradient Boosting clásico de scikit-learn

Al finalizar, el sistema selecciona automáticamente el mejor modelo según ROC-AUC y genera:
- Tabla comparativa de métricas
- Gráficos de comparación
- Matriz de confusión del mejor modelo
- Análisis de eficiencia (velocidad vs performance)

## 📊 Archivos generados

Después de ejecutar, encontrarás:

```
├── models/
│   ├── best_model.pkl                  # Mejor modelo seleccionado
│   ├── best_model_metadata.json        # Metadata del mejor modelo
│   └── xgboost_model.pkl              # Modelo XGBoost específico
├── data/processed/
│   ├── X_train.pkl                     # Datos de entrenamiento
│   ├── X_test.pkl                      # Datos de prueba
│   └── preprocessor.pkl                # Preprocesador
├── outputs/
│   ├── model_comparison.csv            # Comparación de todos los modelos
│   ├── all_models_results.json         # Resultados detallados
│   └── monitoring/
│       ├── drift_results_*.csv         # Resultados de drift
│       ├── predictions_*.csv           # Predicciones
│       └── alerts_*.json               # Alertas generadas
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

### Ver comparación de modelos
El dashboard ahora incluye una nueva sección **"🏆 Comparación de Modelos"** que muestra:
- Tabla comparativa de los 5 modelos entrenados
- Gráficos de comparación de métricas (ROC-AUC, F1, Precision, Recall)
- Análisis de tiempo de entrenamiento
- Matriz de confusión del mejor modelo
- Análisis de eficiencia (performance vs velocidad)

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
