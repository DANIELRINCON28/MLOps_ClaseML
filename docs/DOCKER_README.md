# 🐳 Guía de Ejecución con Docker

Esta guía explica cómo ejecutar **todo el proyecto MLOps** usando Docker, incluyendo la API FastAPI y el Dashboard de Streamlit.

## 📋 Tabla de Contenidos

- [Requisitos Previos](#requisitos-previos)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Inicio Rápido](#inicio-rápido)
- [Gestión de Contenedores](#gestión-de-contenedores)
- [Servicios Disponibles](#servicios-disponibles)
- [Solución de Problemas](#solución-de-problemas)
- [Comandos Útiles](#comandos-útiles)

---

## 📦 Requisitos Previos

### Software Necesario

1. **Docker Desktop** (Windows/Mac) o **Docker Engine** (Linux)
   - Versión mínima: 20.10+
   - Docker Compose v2
   - Descargar: https://www.docker.com/products/docker-desktop

2. **PowerShell** (solo Windows)
   - PowerShell 5.1+ (incluido en Windows 10/11)

### Verificación de Instalación

```powershell
# Verificar Docker
docker --version
docker-compose --version

# Verificar que Docker esté ejecutándose
docker ps
```

### Espacio en Disco

- Espacio mínimo requerido: **5 GB**
- Espacio recomendado: **10 GB**

---

## 🏗️ Arquitectura del Sistema

El proyecto se despliega con **2 contenedores** que se comunican entre sí:

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network (mlops-network)          │
│                                                             │
│  ┌──────────────────────┐      ┌──────────────────────┐   │
│  │   fraud-api          │      │  fraud-dashboard     │   │
│  │   (FastAPI)          │      │  (Streamlit)         │   │
│  │                      │      │                      │   │
│  │  Puerto: 8000        │◄────►│  Puerto: 8501        │   │
│  │                      │      │                      │   │
│  │  - Predicciones      │      │  - Visualizaciones   │   │
│  │  - API REST          │      │  - Monitoreo Drift   │   │
│  │  - Swagger Docs      │      │  - Métricas          │   │
│  └──────────────────────┘      └──────────────────────┘   │
│           │                              │                 │
│           └──────────────┬───────────────┘                 │
│                          │                                 │
│                    ┌─────▼──────┐                         │
│                    │  Volumes   │                         │
│                    │            │                         │
│                    │  - models/ │                         │
│                    │  - data/   │                         │
│                    │  - outputs/│                         │
│                    └────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Componentes

1. **fraud-detection-api**: API REST para predicciones de fraude
2. **fraud-monitoring-dashboard**: Dashboard interactivo para monitoreo
3. **Volúmenes compartidos**: Datos y modelos accesibles desde ambos contenedores

---

## 🚀 Inicio Rápido

### Opción 1: Script Automatizado (Recomendado)

```powershell
# 1. Construir imágenes
.\docker-manager.ps1 build

# 2. Levantar contenedores
.\docker-manager.ps1 up

# 3. Ver estado
.\docker-manager.ps1 status
```

### Opción 2: Comandos Manuales

```powershell
# 1. Navegar al directorio del proyecto
cd "C:\Users\ASUS\Desktop\Final ML\PROYECTO_ML\MLOps_ClaseML"

# 2. Construir las imágenes
cd config
docker-compose build

# 3. Levantar los contenedores
docker-compose up -d

# 4. Verificar que estén ejecutándose
docker-compose ps
```

### Primera Ejecución

La primera vez tomará más tiempo (~5-10 minutos) porque:
- Descarga la imagen base de Python
- Instala todas las dependencias
- Construye las imágenes

**Ejecuciones posteriores son mucho más rápidas** (~30 segundos).

---

## 🎮 Gestión de Contenedores

### Usando el Script PowerShell

```powershell
# Ver todos los comandos disponibles
.\docker-manager.ps1 help

# Construir imágenes
.\docker-manager.ps1 build

# Iniciar servicios
.\docker-manager.ps1 up

# Detener servicios
.\docker-manager.ps1 down

# Reiniciar servicios
.\docker-manager.ps1 restart

# Ver logs en tiempo real
.\docker-manager.ps1 logs

# Ver estado y recursos
.\docker-manager.ps1 status

# Limpiar recursos no usados
.\docker-manager.ps1 clean
```

### Usando Docker Compose Directamente

```powershell
cd config

# Iniciar servicios
docker-compose up -d

# Detener servicios
docker-compose down

# Ver logs
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f fraud-detection-api
docker-compose logs -f fraud-monitoring-dashboard

# Reiniciar un servicio específico
docker-compose restart fraud-detection-api
docker-compose restart fraud-monitoring-dashboard

# Ver estado
docker-compose ps
```

---

## 🌐 Servicios Disponibles

Una vez que los contenedores estén ejecutándose, accede a:

### 🔹 API FastAPI

| Endpoint | URL | Descripción |
|----------|-----|-------------|
| **Página Principal** | http://localhost:8000 | Información general de la API |
| **Swagger UI** | http://localhost:8000/docs | Documentación interactiva |
| **ReDoc** | http://localhost:8000/redoc | Documentación alternativa |
| **Health Check** | http://localhost:8000/health | Estado del servicio |
| **Modelo Info** | http://localhost:8000/model/info | Información del modelo |

#### Ejemplos de Uso

```powershell
# Health Check
curl http://localhost:8000/health

# Información del modelo
curl http://localhost:8000/model/info

# Predicción individual
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C840083671",
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C38997010",
    "oldbalanceDest": 21182.0,
    "newbalanceDest": 0.0
  }'
```

### 🔹 Dashboard Streamlit

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **Dashboard Principal** | http://localhost:8501 | Dashboard interactivo de monitoreo |

#### Funcionalidades del Dashboard

- 📊 **Visualización de Data Drift**: Gráficos de distribuciones
- 🎯 **Métricas del Modelo**: Precisión, Recall, ROC-AUC
- ⚠️ **Alertas**: Notificaciones de drift detectado
- 📈 **Predicciones**: Visualización de resultados
- 📋 **Historial**: Seguimiento de métricas en el tiempo

---

## 🔧 Solución de Problemas

### Problema 1: Puerto en Uso

**Error**: `Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use`

**Solución**:
```powershell
# Opción A: Detener el servicio que usa el puerto
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Opción B: Cambiar el puerto en docker-compose.yml
# Editar: ports: - "8080:8000"  # Usar puerto 8080 en lugar de 8000
```

### Problema 2: Contenedor en Reinicios Constantes

**Diagnóstico**:
```powershell
# Ver estado
docker ps -a | findstr fraud

# Ver logs
docker logs fraud-api
docker logs fraud-dashboard
```

**Soluciones comunes**:
- Verificar que existan los archivos `models/best_model.pkl` y `data/processed/preprocessor.pkl`
- Revisar logs para identificar el error específico
- Reconstruir la imagen: `docker-compose build --no-cache`

### Problema 3: Modelo No Encontrado

**Error en logs**: `ERROR - Modelo no encontrado: models/best_model.pkl`

**Solución**:
```powershell
# Verificar que el modelo existe
ls ..\models\best_model.pkl

# Si no existe, entrenar el modelo primero
python mlops_pipeline\src\model_training_evaluation.py
```

### Problema 4: Error de Memoria

**Error**: `docker: Error response from daemon: failed to create shim: OCI runtime create failed`

**Solución**:
```powershell
# Aumentar memoria asignada a Docker Desktop
# Settings > Resources > Memory > Aumentar a 4GB mínimo
```

### Problema 5: Imágenes Corruptas

**Solución**:
```powershell
# Limpiar todo y reconstruir
docker-compose down
docker system prune -a
docker-compose build --no-cache
docker-compose up -d
```

---

## 📚 Comandos Útiles

### Monitoreo

```powershell
# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de los últimos 100 líneas
docker logs fraud-api --tail 100
docker logs fraud-dashboard --tail 100

# Ver estadísticas de recursos
docker stats fraud-api fraud-dashboard

# Inspeccionar un contenedor
docker inspect fraud-api
```

### Acceso a Contenedores

```powershell
# Entrar a un contenedor (shell interactivo)
docker exec -it fraud-api /bin/bash
docker exec -it fraud-dashboard /bin/bash

# Ejecutar un comando en un contenedor
docker exec fraud-api ls -la /app/models/
docker exec fraud-api python --version
```

### Gestión de Imágenes

```powershell
# Listar imágenes
docker images | findstr fraud

# Eliminar una imagen
docker rmi fraud-detection-api:latest
docker rmi fraud-monitoring-dashboard:latest

# Eliminar imágenes sin usar
docker image prune -a
```

### Gestión de Volúmenes

```powershell
# Listar volúmenes
docker volume ls

# Inspeccionar un volumen
docker volume inspect config_mlops-data

# Eliminar volúmenes sin usar
docker volume prune
```

### Gestión de Redes

```powershell
# Listar redes
docker network ls

# Inspeccionar la red del proyecto
docker network inspect config_mlops-network
```

---

## 🔄 Flujo de Trabajo Típico

### Desarrollo Diario

```powershell
# 1. Levantar servicios
.\docker-manager.ps1 up

# 2. Trabajar con los servicios
# - Abrir http://localhost:8000/docs
# - Abrir http://localhost:8501

# 3. Ver logs si hay problemas
.\docker-manager.ps1 logs

# 4. Detener al finalizar
.\docker-manager.ps1 down
```

### Después de Cambios en el Código

```powershell
# 1. Detener servicios
.\docker-manager.ps1 down

# 2. Reconstruir imágenes
.\docker-manager.ps1 build

# 3. Levantar nuevamente
.\docker-manager.ps1 up

# 4. Verificar que funcionen correctamente
.\docker-manager.ps1 status
```

### Limpieza Periódica

```powershell
# Cada semana/mes, limpiar recursos no usados
.\docker-manager.ps1 clean
```

---

## 📊 Verificación de Funcionamiento

### Checklist de Validación

- [ ] **Contenedores ejecutándose**
  ```powershell
  docker ps | findstr fraud
  # Debe mostrar 2 contenedores: fraud-api y fraud-dashboard
  ```

- [ ] **Health checks pasando**
  ```powershell
  curl http://localhost:8000/health
  # Debe responder: {"status":"healthy","model_loaded":true}
  ```

- [ ] **API accesible**
  - Abrir http://localhost:8000/docs
  - Debe cargar Swagger UI

- [ ] **Dashboard accesible**
  - Abrir http://localhost:8501
  - Debe cargar el dashboard de Streamlit

- [ ] **Logs sin errores críticos**
  ```powershell
  docker logs fraud-api --tail 50
  docker logs fraud-dashboard --tail 50
  ```

---

## 🎓 Universidad Católica Luis Amigó

**Proyecto**: Sistema MLOps de Detección de Fraude  
**Autor**: Daniel Alejandro Rincón Valencia  
**Fecha**: Noviembre 2025  

---

## 📞 Soporte

Si encuentras problemas:

1. Revisa la sección [Solución de Problemas](#solución-de-problemas)
2. Consulta los logs: `docker-compose logs -f`
3. Revisa el archivo `docs/DOCKER_FIX.md` para problemas comunes
4. Verifica que todos los archivos necesarios existan (modelos, datos procesados)

---

## 📄 Licencia

MIT License - Ver archivo LICENSE para más detalles
