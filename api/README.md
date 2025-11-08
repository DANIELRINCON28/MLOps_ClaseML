# ============================================================================
# GUÍA DE USO DE LA API DE DETECCIÓN DE FRAUDE
# ============================================================================

## 📋 Tabla de Contenidos

1. [Instalación](#instalación)
2. [Inicio Rápido](#inicio-rápido)
3. [Endpoints Disponibles](#endpoints-disponibles)
4. [Ejemplos de Uso](#ejemplos-de-uso)
5. [Docker](#docker)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.11 o superior
- Modelo entrenado en `models/best_model.pkl`
- Preprocesador en `data/processed/preprocessor.pkl`

### Instalación de Dependencias

```bash
# Opción 1: Instalar desde requirements.txt del proyecto
pip install -r requirements.txt

# Opción 2: Instalar dependencias específicas de la API
cd api
pip install -r requirements.txt
```

---

## ⚡ Inicio Rápido

### 1. Iniciar el servidor localmente

```bash
# Desde el directorio raíz del proyecto
cd api
python main.py
```

O usando uvicorn directamente:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Acceder a la documentación interactiva

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **API Root:** http://localhost:8000

---

## 📡 Endpoints Disponibles

### 1. **GET /** - Información de la API

```bash
curl http://localhost:8000/
```

**Respuesta:**
```json
{
  "api": "Fraud Detection API",
  "version": "1.0.0",
  "description": "API para detección de fraude en transacciones financieras",
  "endpoints": {...}
}
```

---

### 2. **GET /health** - Estado de Salud

```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-07T14:30:15.123456"
}
```

---

### 3. **GET /model/info** - Información del Modelo

```bash
curl http://localhost:8000/model/info
```

**Respuesta:**
```json
{
  "model_name": "Random_Forest",
  "model_type": "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "roc_auc": 1.0
  },
  "trained_on": "2025-11-07 14:30:15",
  "features_count": 22,
  "status": "loaded"
}
```

---

### 4. **POST /predict** - Predicción Individual

Predice si una transacción individual es fraudulenta.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "PAYMENT",
    "amount": 9839.64,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "nameDest": "M1979787155",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  }'
```

**Respuesta:**
```json
{
  "is_fraud": 0,
  "fraud_probability": 0.023,
  "risk_level": "LOW",
  "timestamp": "2025-11-07T14:30:15.123456",
  "model_version": "Random_Forest"
}
```

**Campos de respuesta:**
- `is_fraud`: 1 = Fraude, 0 = Legítimo
- `fraud_probability`: Probabilidad de fraude (0-1)
- `risk_level`: LOW, MEDIUM, HIGH, CRITICAL
- `timestamp`: Fecha/hora de la predicción
- `model_version`: Versión del modelo usado

---

### 5. **POST /predict/batch** - Predicción por Lotes

Predice múltiples transacciones en una sola solicitud.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
      },
      {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C840083671",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C38997010",
        "oldbalanceDest": 21182.0,
        "newbalanceDest": 0.0
      }
    ]
  }'
```

**Respuesta:**
```json
{
  "predictions": [
    {
      "is_fraud": 0,
      "fraud_probability": 0.023,
      "risk_level": "LOW",
      "timestamp": "2025-11-07T14:30:15.123456",
      "model_version": "Random_Forest"
    },
    {
      "is_fraud": 1,
      "fraud_probability": 0.987,
      "risk_level": "CRITICAL",
      "timestamp": "2025-11-07T14:30:15.123456",
      "model_version": "Random_Forest"
    }
  ],
  "total_transactions": 2,
  "frauds_detected": 1,
  "fraud_rate": 50.0,
  "processing_time_ms": 45.23,
  "timestamp": "2025-11-07T14:30:15.123456",
  "model_version": "Random_Forest"
}
```

---

### 6. **POST /predict/csv** - Predicción desde CSV

Carga un archivo CSV y predice todas las transacciones.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@transactions.csv"
```

**Formato del CSV:**
```csv
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest
1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0
1,TRANSFER,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0
```

**Respuesta:**
```json
{
  "predictions": [...],
  "total_transactions": 100,
  "frauds_detected": 5,
  "fraud_rate": 5.0,
  "processing_time_ms": 234.56,
  "timestamp": "2025-11-07T14:30:15.123456",
  "model_version": "Random_Forest"
}
```

---

## 📝 Ejemplos de Uso

### Python con requests

```python
import requests
import json

# URL base
BASE_URL = "http://localhost:8000"

# 1. Verificar salud de la API
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. Predicción individual
transaction = {
    "step": 1,
    "type": "PAYMENT",
    "amount": 9839.64,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "nameDest": "M1979787155",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=transaction
)

result = response.json()
print(f"Fraude: {result['is_fraud']}")
print(f"Probabilidad: {result['fraud_probability']:.4f}")
print(f"Riesgo: {result['risk_level']}")

# 3. Predicción batch
batch = {
    "transactions": [transaction, transaction]  # Lista de transacciones
}

response = requests.post(
    f"{BASE_URL}/predict/batch",
    json=batch
)

batch_result = response.json()
print(f"Total: {batch_result['total_transactions']}")
print(f"Fraudes: {batch_result['frauds_detected']}")
print(f"Tasa: {batch_result['fraud_rate']:.2f}%")

# 4. Predicción desde CSV
with open('transactions.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        f"{BASE_URL}/predict/csv",
        files=files
    )
    
csv_result = response.json()
print(f"Procesadas: {csv_result['total_transactions']}")
```

### JavaScript/Node.js con fetch

```javascript
const BASE_URL = 'http://localhost:8000';

// Predicción individual
async function predictTransaction(transaction) {
    const response = await fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(transaction)
    });
    
    return await response.json();
}

// Ejemplo de uso
const transaction = {
    step: 1,
    type: "PAYMENT",
    amount: 9839.64,
    nameOrig: "C1231006815",
    oldbalanceOrg: 170136.0,
    newbalanceOrig: 160296.36,
    nameDest: "M1979787155",
    oldbalanceDest: 0.0,
    newbalanceDest: 0.0
};

predictTransaction(transaction)
    .then(result => {
        console.log('Fraude:', result.is_fraud);
        console.log('Probabilidad:', result.fraud_probability);
        console.log('Riesgo:', result.risk_level);
    });
```

### cURL para testing rápido

```bash
# Test básico
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Predicción (Linux/Mac)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @transaction.json

# Predicción (Windows PowerShell)
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body (Get-Content transaction.json)
```

---

## 🐳 Docker

### Construcción de la imagen

```bash
# Desde el directorio raíz del proyecto
docker build -t fraud-detection-api .
```

### Ejecución del contenedor

```bash
# Ejecutar en puerto 8000
docker run -p 8000:8000 fraud-detection-api

# Ejecutar en segundo plano (detached)
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api

# Ver logs
docker logs fraud-api

# Detener contenedor
docker stop fraud-api

# Eliminar contenedor
docker rm fraud-api
```

### Docker Compose (opcional)

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 🔧 Troubleshooting

### Error: "Modelo no disponible"

**Causa:** El modelo no se ha cargado correctamente.

**Solución:**
1. Verificar que existe `models/best_model.pkl`
2. Verificar que existe `data/processed/preprocessor.pkl`
3. Verificar rutas relativas desde `api/main.py`

```bash
# Estructura esperada:
project/
├── api/
│   └── main.py
├── models/
│   └── best_model.pkl
└── data/
    └── processed/
        └── preprocessor.pkl
```

### Error: "ModuleNotFoundError"

**Causa:** Dependencias no instaladas.

**Solución:**
```bash
pip install -r requirements.txt
# o
pip install -r api/requirements.txt
```

### Puerto 8000 ya está en uso

**Solución 1:** Cambiar puerto
```bash
uvicorn api.main:app --port 8001
```

**Solución 2:** Matar proceso que usa el puerto
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Error: "Columnas faltantes en CSV"

**Causa:** El CSV no tiene todas las columnas requeridas.

**Solución:** Asegurar que el CSV tiene estas columnas:
- step
- type
- amount
- nameOrig
- oldbalanceOrg
- newbalanceOrig
- nameDest
- oldbalanceDest
- newbalanceDest

### Predicciones lentas

**Optimizaciones:**
1. Usar predicción batch en lugar de llamadas individuales
2. Aumentar workers de uvicorn:
   ```bash
   uvicorn api.main:app --workers 4
   ```
3. Usar gunicorn + uvicorn para producción:
   ```bash
   gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

---

## 📚 Referencias

- **FastAPI:** https://fastapi.tiangolo.com/
- **Uvicorn:** https://www.uvicorn.org/
- **Pydantic:** https://docs.pydantic.dev/

---

## 👨‍💻 Autor

**DANIEL ALEJANDRO RINCON VALENCIA**  
Universidad Católica Luis Amigó  
Proyecto MLOps - Detección de Fraude

---

## 📄 Licencia

Este proyecto es parte del trabajo final de la asignatura de Machine Learning.
