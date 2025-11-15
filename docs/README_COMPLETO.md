# 🚀 Proyecto MLOps - Detección de Fraude en Transacciones Financieras

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de MLOps para la detección de fraude en transacciones financieras utilizando el dataset PaySim. El pipeline incluye desde la carga de datos hasta el entrenamiento y evaluación de modelos de Machine Learning.

## 📁 Estructura del Proyecto

```
PROYECTO_ML/
├── Base_datos.csv                      # Dataset original
├── config.json                         # Configuración del proyecto
├── requirements.txt                    # Dependencias Python
├── README.md                          # Este archivo
├── INSTRUCCIONES_EJECUCION.md        # Guía de ejecución paso a paso
│
├── data/                              # Datos procesados
│   └── processed/                     # Datos después de cada etapa
│       ├── df_original.pkl
│       ├── df_features_complete.pkl
│       ├── X_train.pkl
│       ├── X_test.pkl
│       ├── y_train.pkl
│       ├── y_test.pkl
│       ├── preprocessor.pkl
│       └── *_metadata.pkl
│
├── models/                            # Modelos entrenados
│   ├── best_model.pkl
│   └── best_model_metadata.json
│
├── outputs/                           # Gráficos y reportes
│   ├── eda_*.png
│   ├── metrics_comparison.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── confusion_matrices.png
│   ├── model_comparison.csv
│   └── evaluation_report.json
│
└── mlops_pipeline/
    └── src/
        ├── Cargar_datos.ipynb                 # 1. Carga de datos
        ├── Comprension_eda.ipynb              # 2. Análisis exploratorio
        ├── Comprension_eda_completo.ipynb     # 2. EDA completo
        ├── ft_engineering.py                  # 3. Feature Engineering
        ├── model_training_evaluation.py       # 4. Entrenamiento y evaluación
        ├── model_deploy.ipynb                 # 5. Despliegue
        ├── model_evaluation.ipynb             # 6. Evaluación adicional
        └── model_monitoring.ipynb             # 7. Monitoreo
```

## 🎯 Objetivos del Proyecto

1. **Detectar transacciones fraudulentas** con alta precisión
2. **Implementar un pipeline MLOps completo** y reproducible
3. **Crear modelos escalables y monitoreables**
4. **Generar insights sobre patrones de fraude**

## 📊 Dataset

**Nombre:** PaySim - Simulación de transacciones móviles de dinero

**Tamaño:** ~200,000 transacciones

**Variables:**
- `step`: Unidad de tiempo (1 step = 1 hora)
- `type`: Tipo de transacción (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- `amount`: Monto de la transacción
- `nameOrig`: ID del cliente origen
- `oldbalanceOrg`: Balance inicial origen
- `newbalanceOrig`: Balance final origen
- `nameDest`: ID del cliente destino
- `oldbalanceDest`: Balance inicial destino
- `newbalanceDest`: Balance final destino
- `isFraud`: **Variable objetivo** (1 = fraude, 0 = legítimo)
- `isFlaggedFraud`: Flag del sistema (no usar en entrenamiento)

**Características del dataset:**
- ⚠️ Altamente desbalanceado (~0.13% fraudes)
- 🎯 Fraude solo ocurre en transacciones TRANSFER y CASH_OUT
- ✅ Sin valores nulos
- ✅ Sin duplicados

## 🔧 Tecnologías Utilizadas

### Python Libraries

```
pandas                # Manipulación de datos
numpy                 # Operaciones numéricas
scikit-learn         # Machine Learning
xgboost              # Gradient Boosting
lightgbm             # Gradient Boosting ligero
imbalanced-learn     # Manejo de clases desbalanceadas
matplotlib           # Visualización
seaborn              # Visualización estadística
```

### Técnicas de ML

- **Balanceo de clases:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Engineering:** Variables derivadas de balances, ratios, temporales
- **Preprocesamiento:** ColumnTransformer con pipelines especializados
- **Modelos evaluados:**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - Gradient Boosting

### Métricas de Evaluación

- **ROC-AUC:** Área bajo la curva ROC
- **PR-AUC:** Precisión-Recall AUC
- **F1-Score:** Media armónica de precisión y recall
- **Precision:** Proporción de predicciones positivas correctas
- **Recall:** Proporción de fraudes detectados
- **Accuracy:** Exactitud general

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd PROYECTO_ML
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar entorno virtual

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 📖 Uso del Pipeline

### Opción 1: Ejecución Completa Automática

Ejecutar todos los scripts en orden:

```bash
# 1. Cargar datos (ejecutar notebook o script)
python -c "import Cargar_datos"

# 2. EDA (ejecutar notebook)
jupyter notebook mlops_pipeline/src/Comprension_eda_completo.ipynb

# 3. Feature Engineering
cd mlops_pipeline/src
python ft_engineering.py

# 4. Model Training & Evaluation
python model_training_evaluation.py
```

### Opción 2: Ejecución Paso a Paso

Ver archivo `INSTRUCCIONES_EJECUCION.md` para una guía detallada paso a paso.

## 📊 Resultados Esperados

Después de ejecutar el pipeline completo, obtendrás:

### Archivos Generados

1. **Datos Procesados:**
   - `data/processed/X_train.pkl` - Features de entrenamiento
   - `data/processed/X_test.pkl` - Features de prueba
   - `data/processed/y_train.pkl` - Target de entrenamiento
   - `data/processed/y_test.pkl` - Target de prueba

2. **Modelos:**
   - `models/best_model.pkl` - Mejor modelo entrenado
   - `models/best_model_metadata.json` - Metadata del modelo

3. **Visualizaciones:**
   - `outputs/eda_*.png` - Gráficos del análisis exploratorio
   - `outputs/metrics_comparison.png` - Comparación de modelos
   - `outputs/roc_curves.png` - Curvas ROC
   - `outputs/pr_curves.png` - Curvas Precision-Recall
   - `outputs/confusion_matrices.png` - Matrices de confusión

4. **Reportes:**
   - `outputs/model_comparison.csv` - Tabla de comparación
   - `outputs/evaluation_report.json` - Reporte completo

### Métricas Típicas

Los modelos bien entrenados deberían alcanzar:

- **ROC-AUC:** > 0.90
- **PR-AUC:** > 0.70
- **F1-Score:** > 0.75
- **Recall:** > 0.80 (importante para fraude)

## 🔍 Componentes del Pipeline

### 1. Cargar_datos.ipynb

**Objetivo:** Carga inicial del dataset y verificación básica

**Salidas:**
- Dataset cargado en memoria
- Información básica del dataset
- Archivo pickle para uso posterior

### 2. Comprension_eda.ipynb / Comprension_eda_completo.ipynb

**Objetivo:** Análisis Exploratorio de Datos exhaustivo

**Incluye:**
- Exploración inicial de datos
- Caracterización de variables
- Análisis univariable (distribuciones, outliers)
- Análisis bivariable (relación con target)
- Análisis multivariable (correlaciones, pairplot)
- Identificación de transformaciones necesarias
- Definición de reglas de validación

**Salidas:**
- Múltiples gráficos de visualización
- Dataset con features identificados
- Resumen del EDA

### 3. ft_engineering.py

**Objetivo:** Ingeniería de características y preprocesamiento

**Pipeline implementado:**

```
ColumnTransformer
├── numeric_transformer (Pipeline)
│   ├── SimpleImputer(strategy='median')
│   └── RobustScaler()
│
└── categoric_transformer (Pipeline)
    ├── SimpleImputer(strategy='most_frequent')
    └── OneHotEncoder(drop='first', handle_unknown='ignore')
```

**Features creadas:**
- Diferencias de balance (origen y destino)
- Errores en balance
- Ratios (amount/balance)
- Features binarios (merchant, balance=0)
- Features temporales (hora, día, weekend, night)
- Features de tipo de transacción
- Categorías de monto

**Salidas:**
- X_train, X_test, y_train, y_test
- Preprocesador ajustado
- Metadata del feature engineering

### 4. model_training_evaluation.py

**Objetivo:** Entrenar y evaluar múltiples modelos de ML

**Modelos entrenados:**
1. Logistic Regression (baseline)
2. Random Forest (ensemble robusto)
3. XGBoost (gradient boosting optimizado)
4. LightGBM (gradient boosting rápido)
5. Gradient Boosting (sklearn)

**Proceso:**
1. Aplicar SMOTE para balanceo
2. Entrenar todos los modelos
3. Evaluar en conjunto de prueba
4. Comparar métricas (performance, consistency, scalability)
5. Seleccionar mejor modelo
6. Generar visualizaciones y reportes

**Salidas:**
- Mejor modelo guardado
- Comparación de modelos (tabla y gráficos)
- Curvas ROC y PR
- Matrices de confusión
- Reporte de evaluación completo

## 📈 Interpretación de Resultados

### Cómo elegir el mejor modelo

Se evalúan 3 criterios principales (según imagen adjunta):

1. **Performance** 🎯
   - ROC-AUC > 0.90
   - PR-AUC > 0.70
   - F1-Score balanceado

2. **Consistency** 🔄
   - Resultados estables entre ejecuciones
   - Bajo overfitting (train vs test)
   - Generalización adecuada

3. **Scalability** ⚡
   - Tiempo de entrenamiento razonable
   - Uso eficiente de memoria
   - Capacidad de procesar datos en producción

### Funciones Auxiliares

El código incluye dos funciones principales:

#### `summarize_classification(results_dict)`

Genera un resumen de los resultados de clasificación en formato tabla.

**Uso:**
```python
from model_training_evaluation import summarize_classification

summary = summarize_classification(trainer.results)
```

#### `build_model(X_train, y_train, model_type='xgboost')`

Construye y entrena un modelo específico.

**Uso:**
```python
from model_training_evaluation import build_model

model = build_model(X_train, y_train, model_type='xgboost')
```

## 🛠️ Personalización

### Modificar hiperparámetros

Edita el método `define_models()` en `model_training_evaluation.py`:

```python
'XGBoost': {
    'model': XGBClassifier(
        n_estimators=200,        # Cambiar aquí
        max_depth=15,            # Cambiar aquí
        learning_rate=0.05,      # Cambiar aquí
        # ...
    ),
    'description': 'Gradient Boosting optimizado'
}
```

### Agregar nuevos modelos

Añade nuevos modelos al diccionario `self.models`:

```python
'Neural_Network': {
    'model': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=42
    ),
    'description': 'Red neuronal multicapa'
}
```

### Crear nuevas features

Edita el método `create_features()` en `ft_engineering.py`:

```python
# Tu nueva feature
self.df_features['mi_nueva_feature'] = (
    # Lógica de la feature
)
```

## 🐛 Troubleshooting

### Error: "Memory Error"

El dataset es grande. Soluciones:

1. Trabajar con una muestra:
```python
df_sample = df.sample(n=50000, random_state=42)
```

2. Usar tipos de datos más eficientes:
```python
df['amount'] = df['amount'].astype('float32')
```

### Error: "SMOTE toma mucho tiempo"

Reducir el sampling_strategy:

```python
trainer.apply_smote(sampling_strategy=0.1)  # En lugar de 0.3
```

### Error: "XGBoost no instalado"

```bash
pip install xgboost
```

### Error: "LightGBM no instalado"

```bash
pip install lightgbm
```

## 📝 Próximos Pasos

1. ✅ Carga de datos
2. ✅ Análisis exploratorio
3. ✅ Feature Engineering
4. ✅ Model Training & Evaluation
5. ⏳ Model Deployment (API con FastAPI)
6. ⏳ Model Monitoring (Streamlit dashboard)
7. ⏳ CI/CD Pipeline (GitHub Actions)

## 👥 Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte de un proyecto académico de MLOps.

## 📧 Contacto

Para preguntas o sugerencias, contactar al equipo de MLOps.

---

**Hecho con ❤️ por el equipo de MLOps**

**Última actualización:** Noviembre 2025
