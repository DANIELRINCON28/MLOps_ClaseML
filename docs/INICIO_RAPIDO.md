# 🚀 INICIO RÁPIDO - MLOps Fraud Detection

## ⚡ Ejecución en 1 Comando

### Windows (PowerShell)

```powershell
.\run_all.ps1
```

### Windows (Git Bash) / Linux / macOS

```bash
./run_all.sh
```

---

## 🐳 Con Docker (Portabilidad Total)

### Construcción y Ejecución

```powershell
# Windows
.\run_all.ps1 -Docker

# Linux/macOS
./run_all.sh --docker
```

### Verificación

```powershell
# Ver contenedores
docker ps

# Ver logs
docker-compose logs -f

# Health check
curl http://localhost:8000/health
```

---

## 📊 Acceso a las Aplicaciones

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **Dashboard** | http://localhost:8501 | Streamlit - Visualización de monitoreo |
| **API Docs** | http://localhost:8000/docs | Swagger UI - Documentación interactiva |
| **API** | http://localhost:8000 | FastAPI - Endpoints de predicción |
| **Health** | http://localhost:8000/health | Estado de la API |

---

## 📦 Exportar e Importar (Transferir a Otro Equipo)

### Exportar

```powershell
# 1. Construir imagen
docker build -t fraud-detection-api:latest .

# 2. Exportar a archivo
docker save fraud-detection-api:latest -o fraud-api.tar

# 3. Comprimir (opcional)
# Windows: Usar 7-Zip o WinRAR
# Linux/macOS:
gzip fraud-api.tar
```

### Importar en Otro Equipo

```powershell
# 1. Copiar fraud-api.tar al nuevo equipo

# 2. Cargar imagen
docker load -i fraud-api.tar

# 3. Ejecutar
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest

# 4. Verificar
# Abrir navegador: http://localhost:8000/docs
```

---

## 🎯 Opciones del Script

### run_all.ps1 / run_all.sh

```powershell
# Ejecución completa (local)
.\run_all.ps1

# Solo API (requiere modelo entrenado)
.\run_all.ps1 -ApiOnly

# Todo con Docker
.\run_all.ps1 -Docker

# Ayuda
.\run_all.ps1 -Help
```

---

## 📁 Estructura de Salidas

```
MLOps_ClaseML/
├── models/
│   ├── best_model.pkl              ← Modelo entrenado
│   └── best_model_metadata.json    ← Métricas y configuración
├── data/processed/
│   ├── preprocessor.pkl            ← Preprocesador
│   └── temp_production_data.csv    ← Datos procesados
└── outputs/
    ├── all_models_results.json     ← Resultados de todos los modelos
    ├── model_comparison.csv        ← Comparación de modelos
    └── monitoring/
        ├── predictions.csv         ← Predicciones
        ├── drift_results_*.csv     ← Detección de drift
        └── alerts_*.json           ← Alertas generadas
```

---

## 🧪 Probar la API

### cURL

```bash
# Predicción individual
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

### PowerShell

```powershell
$body = @{
    step = 1
    type = "PAYMENT"
    amount = 9839.64
    nameOrig = "C1231006815"
    oldbalanceOrg = 170136.0
    newbalanceOrig = 160296.36
    nameDest = "M1979787155"
    oldbalanceDest = 0.0
    newbalanceDest = 0.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

### Python

```python
import requests

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

response = requests.post("http://localhost:8000/predict", json=transaction)
print(response.json())
```

---

## 🛠️ Comandos Útiles

### Docker

```powershell
# Ver imágenes
docker images

# Ver contenedores activos
docker ps

# Ver todos los contenedores
docker ps -a

# Logs en tiempo real
docker-compose logs -f

# Reiniciar contenedor
docker-compose restart

# Detener
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v
```

### Local (sin Docker)

```powershell
# Activar entorno virtual
.\MLOPS_FINAL-venv\Scripts\Activate.ps1  # Windows
source MLOPS_FINAL-venv/bin/activate      # Linux/macOS

# Detener Dashboard
# Buscar PID en streamlit.pid y detener proceso

# Solo API
python -m uvicorn api.main:app --reload
```

---

## 📚 Documentación Completa

- **Docker:** [docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)
- **API:** [api/README.md](api/README.md)
- **Deployment:** [docs/CHECKLIST_DEPLOYMENT.md](docs/CHECKLIST_DEPLOYMENT.md)
- **EDA:** [docs/CHECKLIST_EDA.md](docs/CHECKLIST_EDA.md)
- **Feature Engineering:** [docs/CHECKLIST_FEATURE_ENGINEERING.md](docs/CHECKLIST_FEATURE_ENGINEERING.md)
- **Model Training:** [docs/CHECKLIST_MODEL_TRAINING.md](docs/CHECKLIST_MODEL_TRAINING.md)
- **Monitoring:** [docs/CHECKLIST_DATA_MONITORING.md](docs/CHECKLIST_DATA_MONITORING.md)

---

## ⚠️ Troubleshooting

### Error: "Puerto 8000/8501 en uso"

```powershell
# Windows - Ver qué usa el puerto
netstat -ano | findstr :8000

# Linux/macOS
lsof -i :8000

# Cambiar puerto en Docker
docker run -d -p 8080:8000 --name fraud-api fraud-detection-api:latest
```

### Error: "Python no encontrado"

```powershell
# Verificar instalación
python --version

# O usar python3
python3 --version

# Instalar desde: https://www.python.org/downloads/
```

### Error: "Docker no encontrado"

```powershell
# Instalar Docker Desktop
# Windows/macOS: https://www.docker.com/products/docker-desktop
# Linux: sudo apt-get install docker.io docker-compose

# Verificar
docker --version
```

### Limpiar y reiniciar

```powershell
# Detener todo
docker-compose down

# Limpiar contenedores y cachés
docker system prune -a

# Reconstruir desde cero
docker-compose up --build
```

---

## 🎓 Flujo de Trabajo Recomendado

### Primera Vez (Setup Inicial)

```powershell
# 1. Clonar o descargar el proyecto
git clone <repo-url>
cd MLOps_ClaseML

# 2. Ejecutar pipeline completo
.\run_all.ps1

# 3. Acceder al dashboard
start http://localhost:8501

# 4. (Opcional) Iniciar API
.\run_all.ps1 -ApiOnly
```

### Desarrollo

```powershell
# Ejecutar solo componentes específicos
python mlops_pipeline/src/ft_engineering.py
python mlops_pipeline/src/model_training_evaluation.py
python mlops_pipeline/src/model_monitoring.py
streamlit run app_monitoring.py
```

### Producción (Docker)

```powershell
# 1. Entrenar modelo localmente
.\run_all.ps1

# 2. Construir imagen
docker build -t fraud-api .

# 3. Exportar para distribución
docker save fraud-api -o fraud-api.tar

# 4. En servidor de producción
docker load -i fraud-api.tar
docker run -d -p 8000:8000 --restart always --name fraud-api fraud-api:latest
```

---

## 📞 Soporte

**Documentación adicional:**
- Revisa los archivos en `docs/` para guías detalladas
- Consulta `api/README.md` para ejemplos de uso de la API
- Lee `docs/DOCKER_GUIDE.md` para distribución avanzada

**Verificación del sistema:**
```powershell
# Ejecutar script de verificación
python scripts/check_environment.py
```

---

**Última actualización:** Noviembre 2024  
**Versión:** 1.0.0  
**Proyecto:** MLOps Fraud Detection System
