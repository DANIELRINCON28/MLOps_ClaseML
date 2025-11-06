# 🔍 Sistema de Monitoreo de Data Drift - Guía de Ejecución

## 📋 Descripción

Sistema completo de monitoreo y detección de data drift para el modelo de detección de fraude, incluyendo:

- ✅ Detección automática de data drift (KS, PSI, JS, Chi2)
- ✅ Generación de alertas por severidad
- ✅ Dashboard interactivo con Streamlit
- ✅ Visualización con colores institucionales Universidad Católica Luis Amigó

---

## 🚀 Ejecución Rápida

### Paso 1: Ejecutar el Script de Monitoreo

```powershell
# Navegar al directorio del proyecto
cd c:\Users\Danie\OneDrive\Desktop\ML\PROYECTO_ML\PROYECTO_ML

# Activar entorno virtual
.\MLOPS_FINAL-venv\Scripts\Activate.ps1

# Navegar a la carpeta de scripts
cd mlops_pipeline\src

# Ejecutar monitoreo
python model_monitoring.py
```

**Salida Esperada:**
```
================================================================================
🔍 SISTEMA DE MONITOREO Y DETECCIÓN DE DATA DRIFT
Pipeline MLOps - Detección de Fraude
================================================================================

📂 Cargando datos de referencia...
✅ Datos de referencia cargados: (160000, 20)

🤖 Cargando modelo entrenado...
✅ Modelo y preprocesador cargados

📊 Cargando datos de producción: ...
✅ Datos de producción cargados: (40000, 20)

================================================================================
🔍 INICIANDO DETECCIÓN DE DATA DRIFT
================================================================================

📊 Analizando 15 variables numéricas...

   Analizando: amount
      ✅ KS=0.0456, PSI=0.0389, JS=0.0312

   Analizando: oldbalanceOrg
      ⚠️ KS=0.1234, PSI=0.1456, JS=0.1123

...

================================================================================
🚨 GENERACIÓN DE ALERTAS
================================================================================

⚠️ ADVERTENCIA: 3 variables con drift moderado
   Variables: oldbalanceOrg, amount_to_oldbalance_orig_ratio, balance_diff_orig

📊 RESUMEN GENERAL:
   Total variables analizadas: 15
   Variables con drift: 3
   Severidad alta: 0
   Severidad media: 3

💾 Guardando resultados en ../../outputs/monitoring...
   ✅ Drift results: drift_results_20251106_143022.csv
   ✅ Alerts: alerts_20251106_143022.json
   ✅ Predictions: predictions_20251106_143022.csv
   ✅ Summary: latest_summary.json

✅ Todos los resultados guardados exitosamente

================================================================================
✅ MONITOREO COMPLETADO EXITOSAMENTE
================================================================================

📊 Los resultados están disponibles para visualización en Streamlit
   Ejecuta: streamlit run app_monitoring.py
```

### Paso 2: Ejecutar el Dashboard de Streamlit

```powershell
# Volver al directorio raíz del proyecto
cd ..\..

# Ejecutar aplicación Streamlit
streamlit run app_monitoring.py
```

**Resultado:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

### Paso 3: Explorar el Dashboard

Abre tu navegador en `http://localhost:8501` y explora:

1. **🏠 Resumen General**
   - Métricas clave (Total Variables, Drift Detectado, Severidad)
   - Gráfico de severidad del drift
   - Heatmap de métricas
   - Top 10 variables con mayor drift

2. **📈 Métricas de Drift**
   - Filtros por tipo de variable y severidad
   - Análisis detallado por variable
   - Gauges interactivos (KS, PSI, JS)
   - Estadísticas comparativas

3. **🚨 Alertas y Recomendaciones**
   - Alertas críticas destacadas
   - Recomendaciones automáticas
   - Variables afectadas

4. **🎯 Predicciones del Modelo**
   - Total de predicciones
   - Tasa de fraude detectada
   - Distribución de probabilidades
   - Descarga de reportes CSV

5. **📊 Gráficos EDA**
   - Tabs por categoría (Distribuciones, Boxplots, Correlaciones, etc.)
   - Visualización de gráficos del análisis exploratorio

6. **📋 Tabla de Datos**
   - Explorador completo de datos
   - Descarga de CSV

---

## 📁 Archivos Generados

### Directorio: `outputs/monitoring/`

```
outputs/monitoring/
├── drift_results_YYYYMMDD_HHMMSS.csv     # Resultados de drift por variable
├── alerts_YYYYMMDD_HHMMSS.json           # Alertas generadas
├── predictions_YYYYMMDD_HHMMSS.csv       # Predicciones con datos
└── latest_summary.json                   # Resumen del último monitoreo
```

### Contenido de los Archivos

#### `drift_results_*.csv`

| Columna | Descripción |
|---------|-------------|
| variable | Nombre de la variable |
| tipo | numérica / categórica |
| ks_statistic | Estadístico Kolmogorov-Smirnov |
| ks_p_value | P-value del KS test |
| psi | Population Stability Index |
| js_divergence | Jensen-Shannon Divergence |
| drift_detected | True/False |
| severity | low / medium / high |
| ref_mean | Media en datos de referencia |
| prod_mean | Media en datos de producción |
| mean_change_% | Cambio porcentual en la media |

#### `alerts_*.json`

```json
[
  {
    "timestamp": "2025-11-06T14:30:22",
    "level": "ADVERTENCIA",
    "message": "⚠️ ADVERTENCIA: 3 variables con drift moderado",
    "variables": ["oldbalanceOrg", "amount_to_oldbalance_orig_ratio", "balance_diff_orig"],
    "recommendation": "Monitorear de cerca estas variables en los próximos períodos",
    "details": [...]
  },
  {
    "timestamp": "2025-11-06T14:30:22",
    "level": "INFO",
    "message": "📊 Resumen: 3/15 variables con drift detectado",
    "total_variables": 15,
    "drift_detected": 3,
    "high_severity": 0,
    "medium_severity": 3,
    "recommendation": "Revisar dashboard de monitoreo para más detalles"
  }
]
```

#### `latest_summary.json`

```json
{
  "timestamp": "20251106_143022",
  "total_variables": 15,
  "drift_detected": 3,
  "high_severity": 0,
  "medium_severity": 3,
  "low_severity": 12,
  "predictions": {
    "total": 40000,
    "fraud_detected": 52,
    "fraud_rate": 0.13
  }
}
```

---

## 🎨 Características del Dashboard

### Diseño con Colores Institucionales

El dashboard utiliza la paleta de colores de la **Universidad Católica Luis Amigó**:

- 🔵 **Azul institucional** (#005F9E): Headers, títulos principales
- 🟠 **Naranja** (#FF8C00): Subtítulos, elementos destacados
- 🟢 **Verde** (#28A745): Indicadores positivos, severidad baja
- 🟡 **Amarillo** (#FFC107): Advertencias, severidad media
- 🔴 **Rojo** (#DC3545): Alertas críticas, severidad alta

### Elementos Interactivos

- ✅ **Filtros dinámicos**: Por tipo de variable y severidad
- ✅ **Gauges animados**: Visualización de métricas KS, PSI, JS
- ✅ **Gráficos con Plotly**: Interactivos con zoom y tooltips
- ✅ **Tabs organizados**: Para diferentes categorías de análisis
- ✅ **Descarga de CSV**: Para todos los reportes
- ✅ **Botón de actualización**: Para refrescar datos en tiempo real

---

## 🔄 Configuración del Monitoreo

### Frecuencia de Ejecución

**Recomendado:** Ejecutar el monitoreo cada 24 horas

```powershell
# Opción 1: Manualmente (desarrollo)
python mlops_pipeline/src/model_monitoring.py

# Opción 2: Tarea programada (producción)
# Windows Task Scheduler - ejecutar diariamente a las 2:00 AM
```

### Umbrales de Drift

Los umbrales predeterminados son:

```python
thresholds = {
    'ks_stat': 0.1,      # Kolmogorov-Smirnov
    'psi': 0.2,          # Population Stability Index
    'js_divergence': 0.1, # Jensen-Shannon
    'chi2_pvalue': 0.05   # Chi-cuadrado
}
```

**Interpretación:**

| Métrica | < 0.1 | 0.1 - 0.2 | > 0.2 |
|---------|-------|-----------|-------|
| KS / JS / PSI | ✅ Bajo | ⚠️ Medio | 🚨 Alto |

### Tamaño de Muestra

**Predeterminado:** 5,000 registros

```python
# Para análisis más rápido (desarrollo)
drift_results = monitor.detect_drift(sample_size=1000)

# Para análisis completo (producción)
drift_results = monitor.detect_drift(sample_size=None)  # Todos los datos
```

---

## 🚨 Interpretación de Alertas

### Nivel CRÍTICO 🚨

**Criterios:**
- KS ≥ 0.2 O
- PSI ≥ 0.2 O
- JS ≥ 0.2

**Acción Requerida:**
1. ❗ **Acción inmediata requerida**
2. 🔄 **Considerar reentrenamiento del modelo**
3. 📧 **Notificar al equipo de datos**
4. ⏸️ **Revisar si suspender predicciones automáticas**

### Nivel ADVERTENCIA ⚠️

**Criterios:**
- 0.1 ≤ KS < 0.2 O
- 0.1 ≤ PSI < 0.2 O
- 0.1 ≤ JS < 0.2

**Acción Recomendada:**
1. 👀 **Monitorear de cerca en próximas mediciones**
2. 📝 **Preparar plan de reentrenamiento**
3. 🔍 **Investigar causas del cambio**

### Nivel NORMAL ✅

**Criterios:**
- KS < 0.1 Y
- PSI < 0.1 Y
- JS < 0.1

**Acción:**
- ✅ **Continuar monitoreo regular**
- 📊 **Sin cambios necesarios**

---

## 📊 Métricas de Data Drift Explicadas

### 1. Kolmogorov-Smirnov (KS) Test

**Qué mide:** Máxima diferencia entre funciones de distribución acumulada

**Fórmula:**
```
KS = max|F_reference(x) - F_production(x)|
```

**Ejemplo:**
```python
# Variable: amount
KS = 0.234  # 🚨 Alto
# Interpretación: La distribución de montos cambió significativamente
# Posible causa: Inflación, cambio en comportamiento de usuarios
```

### 2. Population Stability Index (PSI)

**Qué mide:** Cambio en la distribución poblacional

**Fórmula:**
```
PSI = Σ[(actual% - expected%) × ln(actual% / expected%)]
```

**Ejemplo:**
```python
# Variable: oldbalanceOrg
PSI = 0.156  # ⚠️ Medio
# Interpretación: Balance promedio de usuarios cambió moderadamente
# Posible causa: Cambio demográfico, nuevos usuarios
```

### 3. Jensen-Shannon Divergence (JS)

**Qué mide:** Distancia simétrica entre distribuciones

**Fórmula:**
```
JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
```

**Ejemplo:**
```python
# Variable: amount_to_oldbalance_orig_ratio
JS = 0.089  # ✅ Bajo
# Interpretación: El ratio monto/balance es consistente
# Acción: Continuar monitoreando
```

### 4. Chi-Cuadrado (Categóricas)

**Qué mide:** Independencia entre distribuciones categóricas

**Ejemplo:**
```python
# Variable: type (TRANSFER, CASH_OUT, etc.)
Chi2 = 45.67, p-value = 0.0001  # 🚨 Alto
# Interpretación: Distribución de tipos de transacción cambió
# Posible causa: Nuevos servicios, cambio en preferencias
```

---

## 🔧 Troubleshooting

### Problema 1: No se Encuentran Archivos de Monitoreo

**Error:**
```
⚠️ No se encontraron resultados de monitoreo
```

**Solución:**
```powershell
# Ejecutar primero el script de monitoreo
cd mlops_pipeline\src
python model_monitoring.py
```

### Problema 2: Error al Cargar Modelo

**Error:**
```
FileNotFoundError: models/best_model.pkl not found
```

**Solución:**
```powershell
# Entrenar el modelo primero
python model_training_evaluation.py
```

### Problema 3: Streamlit No se Ejecuta

**Error:**
```
streamlit: command not found
```

**Solución:**
```powershell
# Instalar streamlit
pip install streamlit

# O reinstalar todas las dependencias
pip install -r requirements.txt
```

### Problema 4: Puerto 8501 en Uso

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solución:**
```powershell
# Especificar otro puerto
streamlit run app_monitoring.py --server.port 8502
```

---

## 📈 Mejores Prácticas

### 1. Frecuencia de Monitoreo

✅ **Recomendado:** Diario (cada 24 horas)

```python
# Configurar en Windows Task Scheduler
# - Programa: python.exe
# - Argumentos: mlops_pipeline/src/model_monitoring.py
# - Frecuencia: Diaria a las 2:00 AM
```

### 2. Retención de Datos

✅ **Mantener historial de 90 días**

```python
# Limpieza automática de archivos antiguos
import os
from datetime import datetime, timedelta

retention_days = 90
monitoring_dir = Path('outputs/monitoring')

for file in monitoring_dir.glob('drift_results_*.csv'):
    file_date = datetime.strptime(file.stem.split('_')[-2], '%Y%m%d')
    if datetime.now() - file_date > timedelta(days=retention_days):
        file.unlink()  # Eliminar archivo
```

### 3. Alertas Automáticas

✅ **Enviar email cuando drift crítico**

```python
# Agregar al final de model_monitoring.py
if high_severity_count > 0:
    send_email_alert(
        to='equipo-datos@universidad.edu',
        subject='🚨 ALERTA CRÍTICA: Data Drift Detectado',
        body=f'Se detectaron {high_severity_count} variables con drift severo'
    )
```

### 4. Versionado de Resultados

✅ **Incluir timestamp en todos los archivos**

```python
# Ya implementado en model_monitoring.py
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
drift_file = f'drift_results_{timestamp}.csv'
```

---

## 📚 Recursos Adicionales

### Documentación

- **Streamlit:** https://docs.streamlit.io/
- **Plotly:** https://plotly.com/python/
- **SciPy Stats:** https://docs.scipy.org/doc/scipy/reference/stats.html

### Papers

1. "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift" (2019)
2. "A Survey on Concept Drift Adaptation" (2014)
3. "Learning from Imbalanced Data" (2018)

### Tutoriales

- Data Drift Detection: https://www.evidentlyai.com/blog/data-drift-detection
- Streamlit Dashboard: https://streamlit.io/gallery
- Plotly Visualizations: https://plotly.com/python/

---

## ✅ Checklist de Ejecución

### Configuración Inicial

- [x] Entorno virtual activado
- [x] Dependencias instaladas (`streamlit`, `plotly`, etc.)
- [x] Modelo entrenado (`best_model.pkl` existe)
- [x] Preprocesador guardado (`preprocessor.pkl` existe)
- [x] Datos de referencia disponibles

### Ejecución del Monitoreo

- [ ] Ejecutar `model_monitoring.py`
- [ ] Verificar archivos generados en `outputs/monitoring/`
- [ ] Revisar alertas en `alerts_*.json`
- [ ] Validar resumen en `latest_summary.json`

### Visualización en Dashboard

- [ ] Ejecutar `streamlit run app_monitoring.py`
- [ ] Abrir navegador en `http://localhost:8501`
- [ ] Explorar todas las secciones
- [ ] Descargar reportes CSV
- [ ] Tomar acciones según alertas

---

## 🎓 Caso de Uso Ejemplo

### Escenario: Detección de Cambio en Comportamiento de Usuarios

**Situación:**
- El modelo fue entrenado con datos de enero-marzo 2025
- Estamos monitoreando datos de noviembre 2025
- Hipótesis: Campaña de marketing cambió el comportamiento

**Ejecución:**

1. **Ejecutar Monitoreo:**
```powershell
python mlops_pipeline/src/model_monitoring.py
```

2. **Revisar Dashboard:**
```
🚨 ALERTA: Variable 'amount' con PSI = 0.267
   - Media Referencia: $179,863
   - Media Producción: $215,836
   - Cambio: +20.0%
```

3. **Análisis:**
   - Los montos promedio aumentaron 20%
   - Posible causa: Campaña de descuentos en transacciones grandes
   - Impacto: Modelo puede sub-detectar fraudes en rangos altos

4. **Acción:**
   - ✅ Reentrenar modelo con datos recientes
   - ✅ Ajustar umbrales de detección
   - ✅ Monitorear performance del modelo
   - ✅ Documentar el cambio

**Resultado:**
- Modelo reentrenado con ROC-AUC mejorado de 95.23% a 96.12%
- Detección de fraude en rangos altos mejoró 15%
- Sistema automatizado detectó el problema antes de impacto severo

---

**🎓 Universidad Católica Luis Amigó - Pipeline MLOps - 2025**

**Desarrollado con ❤️ para mejorar la detección de fraude**
