# 🚀 GUÍA RÁPIDA DE EJECUCIÓN - Sistema de Monitoreo

## ⚡ Ejecución en 3 Pasos

### 📦 Paso 1: Preparar Entorno

```powershell
# Navegar al proyecto
cd c:\Users\Danie\OneDrive\Desktop\ML\PROYECTO_ML\PROYECTO_ML

# Activar entorno virtual
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\MLOPS_FINAL-venv\Scripts\Activate.ps1

# Verificar instalación
python check_environment.py
```

---

### 🔍 Paso 2: Ejecutar Monitoreo de Data Drift

```powershell
# Navegar a la carpeta de scripts
cd mlops_pipeline\src

# Ejecutar sistema de monitoreo
python model_monitoring.py
```

**⏱️ Tiempo estimado:** 2-3 minutos

**✅ Salida esperada:**
- Archivos en `outputs/monitoring/`:
  - `drift_results_YYYYMMDD_HHMMSS.csv`
  - `alerts_YYYYMMDD_HHMMSS.json`
  - `predictions_YYYYMMDD_HHMMSS.csv`
  - `latest_summary.json`

---

### 📊 Paso 3: Abrir Dashboard Interactivo

```powershell
# Volver al directorio raíz
cd ..\..

# Ejecutar aplicación Streamlit
streamlit run app_monitoring.py
```

**🌐 URL del Dashboard:**
- Local: `http://localhost:8501`

**🎨 Características:**
- ✅ Colores institucionales Universidad Católica Luis Amigó
- ✅ 6 secciones interactivas
- ✅ Filtros dinámicos
- ✅ Gráficos con Plotly
- ✅ Descarga de reportes CSV

---

## 📋 Secciones del Dashboard

| Sección | Descripción | Funcionalidades |
|---------|-------------|-----------------|
| 🏠 **Resumen General** | Vista general del estado del sistema | Métricas clave, gráfico de severidad, top 10 variables |
| 📈 **Métricas de Drift** | Análisis detallado por variable | Gauges KS/PSI/JS, filtros, estadísticas comparativas |
| 🚨 **Alertas** | Alertas automáticas por severidad | Alertas críticas, advertencias, recomendaciones |
| 🎯 **Predicciones** | Resultados del modelo | Distribución, tasa de fraude, descarga CSV |
| 📊 **Gráficos EDA** | Visualizaciones del análisis exploratorio | Tabs por categoría, imágenes interactivas |
| 📋 **Tabla de Datos** | Explorador de datos completo | Drift results, predicciones, alertas en tabla |

---

## 🎯 Interpretación Rápida de Alertas

### 🚨 CRÍTICO (Rojo)
- **Acción:** Revisar inmediatamente
- **Recomendación:** Considerar reentrenamiento
- **Criterio:** KS/PSI/JS ≥ 0.2

### ⚠️ ADVERTENCIA (Amarillo)
- **Acción:** Monitorear de cerca
- **Recomendación:** Preparar plan de reentrenamiento
- **Criterio:** 0.1 ≤ KS/PSI/JS < 0.2

### ✅ NORMAL (Verde)
- **Acción:** Continuar monitoreo regular
- **Recomendación:** Sin cambios necesarios
- **Criterio:** KS/PSI/JS < 0.1

---

## 🔄 Comandos Útiles

### Detener el Dashboard
```powershell
# Presionar Ctrl + C en la terminal donde corre Streamlit
```

### Cambiar Puerto del Dashboard
```powershell
streamlit run app_monitoring.py --server.port 8502
```

### Ver Logs del Monitoreo
```powershell
# Los logs se muestran directamente en la consola durante la ejecución
python model_monitoring.py > logs_monitoreo.txt 2>&1
```

### Limpiar Archivos Antiguos
```powershell
# Eliminar archivos de monitoreo de más de 30 días
cd outputs\monitoring
Get-ChildItem -Filter "drift_results_*.csv" | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item
```

---

## 📁 Estructura de Archivos Generados

```
PROYECTO_ML/
├── outputs/
│   ├── monitoring/
│   │   ├── drift_results_20251106_143022.csv     ← Métricas de drift
│   │   ├── alerts_20251106_143022.json           ← Alertas generadas
│   │   ├── predictions_20251106_143022.csv       ← Predicciones
│   │   └── latest_summary.json                   ← Resumen último monitoreo
│   │
│   └── eda_*.png                                  ← Gráficos del EDA
│
├── models/
│   ├── best_model.pkl                             ← Modelo entrenado
│   └── best_model_metadata.json                   ← Metadata del modelo
│
├── data/
│   └── processed/
│       ├── X_train.pkl, X_test.pkl               ← Datos procesados
│       └── preprocessor.pkl                       ← Pipeline de transformación
│
├── mlops_pipeline/
│   └── src/
│       ├── model_monitoring.py                    ← Script de monitoreo
│       ├── Cargar_datos.ipynb                     ← Notebook de carga
│       ├── Comprension_eda_completo.ipynb         ← Notebook de EDA
│       ├── ft_engineering.py                      ← Feature engineering
│       └── model_training_evaluation.py           ← Entrenamiento
│
└── app_monitoring.py                              ← Dashboard Streamlit
```

---

## ⚙️ Configuración Avanzada

### Modificar Umbrales de Drift

Editar `mlops_pipeline/src/model_monitoring.py`:

```python
# Línea ~41
self.thresholds = {
    'ks_stat': 0.1,      # Kolmogorov-Smirnov
    'psi': 0.2,          # Population Stability Index
    'js_divergence': 0.1, # Jensen-Shannon
    'chi2_pvalue': 0.05   # Chi-cuadrado
}

# Valores más estrictos (detecta cambios más pequeños):
self.thresholds = {
    'ks_stat': 0.05,
    'psi': 0.1,
    'js_divergence': 0.05,
    'chi2_pvalue': 0.05
}

# Valores más permisivos (menos alertas):
self.thresholds = {
    'ks_stat': 0.15,
    'psi': 0.25,
    'js_divergence': 0.15,
    'chi2_pvalue': 0.05
}
```

### Modificar Tamaño de Muestra

```python
# En model_monitoring.py, línea ~608

# Análisis rápido (desarrollo)
drift_results = monitor.detect_drift(sample_size=1000)

# Análisis balanceado (recomendado)
drift_results = monitor.detect_drift(sample_size=5000)

# Análisis completo (producción)
drift_results = monitor.detect_drift(sample_size=None)
```

---

## 🎓 Caso de Uso Completo

### Escenario: Monitoreo Semanal

**Lunes 9:00 AM - Ejecutar Monitoreo:**

```powershell
# 1. Activar entorno
.\MLOPS_FINAL-venv\Scripts\Activate.ps1

# 2. Ejecutar monitoreo
cd mlops_pipeline\src
python model_monitoring.py

# 3. Revisar salida
# Si hay alertas críticas → notificar al equipo
# Si hay advertencias → agendar revisión
```

**Salida Ejemplo:**
```
🚨 ALERTA CRÍTICA: 2 variables con drift severo detectado
Variables: amount, oldbalanceOrg
Recomendación: ACCIÓN INMEDIATA REQUERIDA - Considerar reentrenamiento

⚠️ ADVERTENCIA: 1 variables con drift moderado
Variables: balance_diff_orig
Recomendación: Monitorear de cerca estas variables
```

**Acción Tomada:**
1. ✅ Abrir dashboard para análisis detallado
2. ✅ Revisar distribuciones de `amount` y `oldbalanceOrg`
3. ✅ Investigar causa (cambio en comportamiento de usuarios, inflación, etc.)
4. ✅ Decidir: ¿Reentrenar modelo? ¿Ajustar umbrales?
5. ✅ Documentar decisión en bitácora

---

## 🆘 Troubleshooting Rápido

### Problema: "No module named 'streamlit'"
**Solución:**
```powershell
pip install streamlit plotly
```

### Problema: "No se encontraron archivos de monitoreo"
**Solución:**
```powershell
cd mlops_pipeline\src
python model_monitoring.py
```

### Problema: "FileNotFoundError: best_model.pkl"
**Solución:**
```powershell
# Entrenar el modelo primero
cd mlops_pipeline\src
python model_training_evaluation.py
```

### Problema: Puerto 8501 en uso
**Solución:**
```powershell
streamlit run app_monitoring.py --server.port 8502
```

### Problema: Dashboard no carga gráficos EDA
**Solución:**
```powershell
# Ejecutar el notebook de EDA primero
jupyter notebook mlops_pipeline/src/Comprension_eda_completo.ipynb
# Ejecutar todas las celdas
```

---

## 📧 Soporte

**Proyecto:** Pipeline MLOps - Detección de Fraude  
**Universidad:** Católica Luis Amigó  
**Fecha:** Noviembre 2025

**Documentación Completa:**
- 📄 `INSIGHTS.md` - Caso de negocio y hallazgos
- 📖 `README_MONITOREO.md` - Guía detallada de monitoreo
- 📋 `README_COMPLETO.md` - Documentación general del proyecto
- 🚀 `INSTRUCCIONES_EJECUCION.md` - Guía paso a paso completa

---

**🎯 ¡Listo para detectar data drift y mantener tu modelo de fraude en óptimas condiciones!**

🔵 **Universidad Católica Luis Amigó** | 🟠 **MLOps** | 🔍 **Fraud Detection**
