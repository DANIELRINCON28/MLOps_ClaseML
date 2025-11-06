# 📖 Instrucciones de Ejecución - Pipeline MLOps de Detección de Fraude

## 🎯 Objetivo

Este documento proporciona una guía paso a paso para ejecutar el pipeline completo de MLOps para detección de fraude.

## ⏱️ Tiempo Estimado

- **Setup inicial:** 10 minutos
- **Ejecución completa:** 30-60 minutos (dependiendo del hardware)

---

## 📋 Paso 0: Preparación del Entorno

### 0.1 Verificar Python

Abre PowerShell y verifica la versión de Python:

```powershell
python --version
```

Debe ser Python 3.8 o superior.

### 0.2 Navegar al directorio del proyecto

```powershell
cd C:\Users\Danie\OneDrive\Desktop\ML\PROYECTO_ML\PROYECTO_ML
```

### 0.3 Crear y activar entorno virtual

```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\Activate.ps1
```

Si PowerShell da error de permisos, ejecuta:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 0.4 Instalar dependencias

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Tiempo estimado:** 5-10 minutos

### 0.5 Verificar instalación

```powershell
python -c "import pandas, sklearn, xgboost, lightgbm, imblearn; print('✅ Todas las librerías instaladas')"
```

---

## 📊 Paso 1: Cargar Datos

### 1.1 Ejecutar notebook de carga de datos

**Opción A: Usando Jupyter Notebook**

```powershell
jupyter notebook
```

Luego navega a: `mlops_pipeline/src/Cargar_datos.ipynb`

Ejecuta todas las celdas (Cell > Run All)

**Opción B: Usando VS Code**

1. Abre VS Code en el directorio del proyecto
2. Abre `mlops_pipeline/src/Cargar_datos.ipynb`
3. Selecciona el kernel de Python (venv)
4. Ejecuta todas las celdas

### 1.2 Verificar salidas

Deberías ver:
- ✅ Datos cargados: ~200,000 filas x 11 columnas
- ✅ No hay valores nulos
- ✅ Distribución de fraude mostrada

### 1.3 Archivos generados

```
data/processed/
├── df_original.pkl
└── dataset_info.pkl
```

**Tiempo estimado:** 2-3 minutos

---

## 🔍 Paso 2: Análisis Exploratorio de Datos (EDA)

### 2.1 Ejecutar notebook de EDA completo

**Opción A: Jupyter Notebook**

```powershell
jupyter notebook mlops_pipeline/src/Comprension_eda_completo.ipynb
```

**Opción B: VS Code**

Abre `Comprension_eda_completo.ipynb` y ejecuta todas las celdas.

### 2.2 Revisión de análisis

El notebook realizará:

1. **Exploración inicial**
   - Vista general de datos
   - Información de tipos de datos
   - Análisis de nulos y duplicados

2. **Caracterización de variables**
   - Variables numéricas vs categóricas
   - Variables binarias (target)

3. **Análisis univariable**
   - Distribuciones de variables numéricas
   - Gráficos de barras para categóricas
   - Detección de outliers

4. **Análisis bivariable**
   - Fraude por tipo de transacción
   - Comparación de montos
   - Análisis temporal

5. **Análisis multivariable**
   - Matriz de correlación
   - Pairplot de variables clave

6. **Identificación de features**
   - Features derivados de balances
   - Features binarios
   - Features de ratios
   - Features temporales

### 2.3 Archivos generados

```
data/processed/
├── df_eda.pkl
├── df_features.pkl
└── eda_summary.pkl

outputs/
├── eda_distribucion_numericas.png
├── eda_boxplots_numericas.png
├── eda_categoricas.png
├── eda_fraude_por_tipo.png
├── eda_montos_fraude.png
├── eda_temporal_fraude.png
├── eda_correlacion.png
└── eda_pairplot.png
```

**Tiempo estimado:** 10-15 minutos

### 2.4 Hallazgos clave a observar

- ⚠️ Dataset desbalanceado (~0.13% fraudes)
- 🎯 Fraude SOLO en transacciones TRANSFER y CASH_OUT
- 📊 Diferencias en montos entre fraude y no fraude
- 🔍 Patrones en balances de cuentas fraudulentas

---

## 🔧 Paso 3: Feature Engineering

### 3.1 Ejecutar script de feature engineering

```powershell
cd mlops_pipeline\src
python ft_engineering.py
```

### 3.2 Observar la ejecución

El script realizará:

```
1. Cargando datos...
   ✅ Datos cargados: ~200,000 filas x 11 columnas

2. Creando nuevas características...
   📊 Creando features de balance...
   📊 Creando features binarios...
   📊 Creando features de ratios...
   📊 Creando features temporales...
   📊 Creando features de tipo de transacción...
   📊 Creando features de magnitud...
   ✅ 16 nuevas features creadas

3. Preparando datos para modelado...
   📊 Features (X): (200000, 26)
   🎯 Target (y): (200000,)
   📊 Distribución de clases:
      - No Fraude: 199,863 (99.87%)
      - Fraude: 263 (0.13%)
   ✅ División completada:
      📊 Train: 160,000 muestras
      📊 Test: 40,000 muestras

4. Construyendo pipeline de preprocesamiento...
   📊 Variables numéricas: 24
   📊 Variables categóricas: 2
   ✅ Pipeline de preprocesamiento construido

5. Ajustando y transformando datos...
   ✅ Datos transformados
   📊 X_train procesado: (160000, 28)
   📊 X_test procesado: (40000, 28)

6. Guardando artefactos...
   ✅ Datasets guardados
   ✅ Preprocesador guardado
   ✅ Dataset completo guardado
   ✅ Metadatos guardados

FEATURE ENGINEERING COMPLETADO ✅
```

### 3.3 Archivos generados

```
data/processed/
├── X_train.pkl                          # Features de entrenamiento
├── X_test.pkl                           # Features de prueba
├── y_train.pkl                          # Target de entrenamiento
├── y_test.pkl                           # Target de prueba
├── preprocessor.pkl                     # Pipeline de preprocesamiento
├── df_features_complete.pkl             # Dataset completo con features
└── feature_engineering_metadata.pkl     # Metadata
```

**Tiempo estimado:** 3-5 minutos

---

## 🤖 Paso 4: Entrenamiento y Evaluación de Modelos

### 4.1 Ejecutar script de training

```powershell
# Asegúrate de estar en mlops_pipeline/src
python model_training_evaluation.py
```

### 4.2 Observar la ejecución

El script realizará:

```
1. Cargando datos preprocesados...
   ✅ Datos cargados
   📊 X_train: (160000, 28)
   📊 X_test: (40000, 28)
   🎯 y_train: (160000,) - Fraude: 263
   🎯 y_test: (40000,) - Fraude: 37

2. Aplicando SMOTE (sampling_strategy=0.3)...
   Antes - Clase 0: 159,737, Clase 1: 263
   Después - Clase 0: 159,737, Clase 1: 47,921
   ✅ SMOTE aplicado

3. Definiendo modelos...
   ✅ 5 modelos definidos:
   • Logistic_Regression: Modelo lineal simple y interpretable
   • Random_Forest: Ensemble de árboles de decisión
   • XGBoost: Gradient Boosting optimizado
   • LightGBM: Gradient Boosting ligero y rápido
   • Gradient_Boosting: Gradient Boosting de sklearn

4. Entrenando modelos...
   🔄 Entrenando Logistic_Regression...
   ✅ Logistic_Regression entrenado en 12.45 segundos
   
   🔄 Entrenando Random_Forest...
   ✅ Random_Forest entrenado en 45.67 segundos
   
   🔄 Entrenando XGBoost...
   ✅ XGBoost entrenado en 23.89 segundos
   
   🔄 Entrenando LightGBM...
   ✅ LightGBM entrenado en 15.34 segundos
   
   🔄 Entrenando Gradient_Boosting...
   ✅ Gradient_Boosting entrenado en 89.12 segundos
   
   ✅ Todos los modelos entrenados

5. Evaluando modelos...
   [Métricas para cada modelo]

6. Comparando modelos...
   [Tabla de comparación]
   [Generando gráficos...]

7. Seleccionando mejor modelo (criterio: roc_auc)...
   🥇 MEJOR MODELO: XGBoost
   Score (roc_auc): 0.9534

8. Generando reporte completo...
   ✅ Reporte guardado

9. Guardando mejor modelo...
   ✅ Modelo guardado en models/best_model.pkl
   ✅ Metadata guardado en models/best_model_metadata.json

MODEL TRAINING & EVALUATION COMPLETADO ✅
```

### 4.3 Archivos generados

```
models/
├── best_model.pkl                    # Mejor modelo entrenado
└── best_model_metadata.json          # Metadata del modelo

outputs/
├── model_comparison.csv              # Tabla de comparación
├── evaluation_report.json            # Reporte JSON
├── metrics_comparison.png            # Gráfico de métricas
├── roc_curves.png                    # Curvas ROC
├── pr_curves.png                     # Curvas Precision-Recall
└── confusion_matrices.png            # Matrices de confusión
```

**Tiempo estimado:** 5-10 minutos (dependiendo del hardware)

### 4.4 Revisar resultados

#### Ver tabla de comparación

```powershell
# Ver en pandas
python -c "import pandas as pd; df = pd.read_csv('../../outputs/model_comparison.csv'); print(df)"
```

#### Ver reporte JSON

```powershell
# Ver en consola
type ..\..\outputs\evaluation_report.json
```

#### Abrir gráficos

Navega a la carpeta `outputs/` y abre los archivos PNG:

- `metrics_comparison.png` - Comparación de todas las métricas
- `roc_curves.png` - Curvas ROC de todos los modelos
- `pr_curves.png` - Curvas Precision-Recall
- `confusion_matrices.png` - Matrices de confusión

---

## 📊 Paso 5: Interpretación de Resultados

### 5.1 Revisar métricas del mejor modelo

El mejor modelo (usualmente XGBoost o LightGBM) debería tener:

- **ROC-AUC:** > 0.90 ✅
- **PR-AUC:** > 0.70 ✅
- **F1-Score:** > 0.75 ✅
- **Recall:** > 0.80 ✅ (crucial para fraude)
- **Precision:** > 0.70 ✅

### 5.2 Analizar curvas ROC

- Curva más alejada de la diagonal = mejor modelo
- Área bajo la curva (AUC) más cercana a 1.0 = mejor

### 5.3 Analizar Precision-Recall

- Importante para datasets desbalanceados
- Muestra el trade-off entre precisión y recall
- AUC > 0.70 es excelente para fraude

### 5.4 Interpretar matriz de confusión

```
                 Predicted
                 No    Fraude
Actual No       [TN]   [FP]
       Fraude   [FN]   [TP]
```

**Métricas importantes:**
- **True Positives (TP):** Fraudes correctamente detectados
- **False Negatives (FN):** Fraudes NO detectados ❌ (minimizar)
- **False Positives (FP):** Falsos positivos (minimizar)
- **True Negatives (TN):** No fraudes correctamente clasificados

---

## 🎯 Criterios de Selección del Mejor Modelo

Según la imagen adjunta, se evalúan 3 aspectos:

### 1. **Performance** 🎯

- ROC-AUC > 0.90
- PR-AUC > 0.70
- F1-Score balanceado
- Recall alto (detectar la mayoría de fraudes)

### 2. **Consistency** 🔄

- Resultados estables
- No overfitting (train vs test similar)
- Generalización adecuada

### 3. **Scalability** ⚡

- Tiempo de entrenamiento < 2 minutos
- Uso de memoria razonable
- Capacidad de procesar en tiempo real

**Modelo típicamente seleccionado:** XGBoost o LightGBM

---

## ✅ Verificación Final

### Checklist de archivos generados

Verifica que existan los siguientes archivos:

```powershell
# Datos
dir data\processed\

# Modelos
dir models\

# Outputs
dir outputs\
```

Deberías tener:

```
✅ data/processed/
   ✅ df_original.pkl
   ✅ X_train.pkl
   ✅ X_test.pkl
   ✅ y_train.pkl
   ✅ y_test.pkl
   ✅ preprocessor.pkl
   ✅ df_features_complete.pkl
   ✅ *_metadata.pkl

✅ models/
   ✅ best_model.pkl
   ✅ best_model_metadata.json

✅ outputs/
   ✅ eda_*.png (8 gráficos)
   ✅ metrics_comparison.png
   ✅ roc_curves.png
   ✅ pr_curves.png
   ✅ confusion_matrices.png
   ✅ model_comparison.csv
   ✅ evaluation_report.json
```

---

## 🔄 Uso del Modelo Entrenado

### Cargar el mejor modelo

```python
import pickle

# Cargar modelo
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar preprocessor
with open('data/processed/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Cargar datos de prueba
import pandas as pd
X_test = pd.read_pickle('data/processed/X_test.pkl')
y_test = pd.read_pickle('data/processed/y_test.pkl')

# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"Predicciones realizadas: {len(y_pred)}")
print(f"Fraudes detectados: {y_pred.sum()}")
```

### Predecir en nuevos datos

```python
# Cargar nuevos datos
new_data = pd.read_csv('nuevos_datos.csv')

# Aplicar feature engineering (usar las mismas transformaciones)
# ... (aplicar las mismas transformaciones de ft_engineering.py)

# Preprocesar
new_data_processed = preprocessor.transform(new_data)

# Predecir
predictions = model.predict(new_data_processed)
probabilities = model.predict_proba(new_data_processed)[:, 1]

# Transacciones con alta probabilidad de fraude
fraud_threshold = 0.5
high_risk = probabilities > fraud_threshold

print(f"Transacciones de alto riesgo: {high_risk.sum()}")
```

---

## 🐛 Solución de Problemas

### Problema 1: "Memory Error"

**Solución:** Trabajar con una muestra más pequeña

```python
# En ft_engineering.py, modificar load_data():
df_sample = self.df.sample(n=50000, random_state=42)
self.df = df_sample
```

### Problema 2: SMOTE toma mucho tiempo

**Solución:** Reducir sampling_strategy

```python
# En model_training_evaluation.py:
trainer.apply_smote(sampling_strategy=0.1)  # En lugar de 0.3
```

### Problema 3: Error al importar librerías

**Solución:** Reinstalar dependencias

```powershell
pip uninstall -y scikit-learn xgboost lightgbm imbalanced-learn
pip install scikit-learn xgboost lightgbm imbalanced-learn
```

### Problema 4: Jupyter no encuentra el kernel

**Solución:** Instalar kernel de IPython

```powershell
pip install ipykernel
python -m ipykernel install --user --name=venv
```

---

## 📚 Recursos Adicionales

### Documentación

- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Imbalanced-learn](https://imbalanced-learn.org/)

### Conceptos clave

- **SMOTE:** Synthetic Minority Over-sampling Technique
- **ROC-AUC:** Receiver Operating Characteristic - Area Under Curve
- **PR-AUC:** Precision-Recall Area Under Curve
- **ColumnTransformer:** Scikit-learn transformer for different column types

---

## 📞 Soporte

Si encuentras problemas:

1. Revisa esta guía completa
2. Consulta el archivo `README_COMPLETO.md`
3. Revisa los logs de error
4. Contacta al equipo de MLOps

---

**¡Felicidades! Has completado el pipeline MLOps de detección de fraude** 🎉

**Siguiente paso:** Implementar el despliegue del modelo (API con FastAPI) y monitoreo (Streamlit).

---

**Última actualización:** Noviembre 2025
