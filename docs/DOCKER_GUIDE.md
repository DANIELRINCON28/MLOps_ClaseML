# 🐳 GUÍA COMPLETA DE DOCKER - MLOps Fraud Detection

## 📋 Índice

1. [Introducción](#introducción)
2. [Instalación de Docker](#instalación-de-docker)
3. [Construcción de la Imagen](#construcción-de-la-imagen)
4. [Ejecución del Contenedor](#ejecución-del-contenedor)
5. [Exportar e Importar Imágenes](#exportar-e-importar-imágenes)
6. [Distribución a Otros Equipos](#distribución-a-otros-equipos)
7. [Docker Hub (Opcional)](#docker-hub-opcional)
8. [Troubleshooting](#troubleshooting)

---

## 📖 Introducción

Esta guía te permite ejecutar el proyecto completo de MLOps en **cualquier equipo** usando Docker, garantizando:

✅ **Portabilidad total** - Funciona en Windows, Linux y macOS  
✅ **Independencia del sistema** - No requiere Python instalado  
✅ **Reproducibilidad** - Mismo entorno en todos los equipos  
✅ **Facilidad de distribución** - Una imagen, múltiples máquinas  

---

## 🔧 Instalación de Docker

### Windows

1. **Descargar Docker Desktop:**
   - Ir a: https://www.docker.com/products/docker-desktop
   - Descargar e instalar Docker Desktop para Windows
   - Reiniciar el equipo si se solicita

2. **Verificar instalación:**
   ```powershell
   docker --version
   docker-compose --version
   ```

3. **Configuración recomendada:**
   - Abrir Docker Desktop
   - Ir a Settings → Resources → Advanced
   - Asignar al menos:
     - **CPU:** 2 cores
     - **Memory:** 4 GB
     - **Disk:** 20 GB

### Linux (Ubuntu/Debian)

```bash
# Actualizar paquetes
sudo apt-get update

# Instalar dependencias
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Agregar clave GPG oficial de Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Configurar repositorio
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verificar
docker --version
docker compose version
```

### macOS

1. **Descargar Docker Desktop:**
   - Ir a: https://www.docker.com/products/docker-desktop
   - Descargar para macOS (Intel o Apple Silicon)
   - Instalar arrastrando a Applications

2. **Verificar:**
   ```bash
   docker --version
   docker compose version
   ```

---

## 🏗️ Construcción de la Imagen

### Paso 1: Preparar el proyecto

Antes de construir la imagen Docker, asegúrate de tener el modelo entrenado:

**Opción A: Entrenar localmente primero (recomendado)**

```powershell
# Windows PowerShell
.\run_all.ps1

# Windows Git Bash / Linux / macOS
./run_all.sh
```

Esto genera:
- `models/best_model.pkl` - Modelo entrenado
- `data/processed/preprocessor.pkl` - Preprocesador
- `models/best_model_metadata.json` - Metadatos

**Opción B: Dejar que Docker lo haga**

El script `run_all.ps1 -Docker` puede entrenar automáticamente si no existe el modelo.

### Paso 2: Construir la imagen

#### Usando Docker Compose (Recomendado)

```powershell
# Construir la imagen
docker-compose build

# O construir y ejecutar en un solo paso
docker-compose up --build -d
```

#### Usando Docker directamente

```powershell
# Construcción básica
docker build -t fraud-detection-api:latest .

# Construcción con tag específico
docker build -t fraud-detection-api:1.0.0 .

# Sin usar cache (reconstrucción limpia)
docker build --no-cache -t fraud-detection-api:latest .
```

**Tiempo estimado:** 5-10 minutos (primera vez)

### Paso 3: Verificar la imagen

```powershell
# Listar imágenes
docker images

# Deberías ver algo como:
# REPOSITORY              TAG       IMAGE ID       CREATED         SIZE
# fraud-detection-api     latest    abc123def456   2 minutes ago   ~600MB
```

---

## 🚀 Ejecución del Contenedor

### Método 1: Docker Compose (Recomendado)

```powershell
# Iniciar contenedor
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener contenedor
docker-compose down

# Reiniciar
docker-compose restart
```

### Método 2: Docker run

```powershell
# Ejecutar contenedor
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  --restart unless-stopped \
  fraud-detection-api:latest

# Ver logs
docker logs -f fraud-api

# Detener
docker stop fraud-api

# Iniciar nuevamente
docker start fraud-api

# Eliminar contenedor
docker rm -f fraud-api
```

### Verificar que funciona

Abre tu navegador y visita:

- **API:** http://localhost:8000
- **Documentación:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

O usa curl:

```powershell
# Health check
curl http://localhost:8000/health

# Predicción de prueba
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
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

---

## 📦 Exportar e Importar Imágenes

### Exportar imagen a archivo

Esta es la forma de compartir tu imagen sin usar Docker Hub.

#### Exportar

```powershell
# Exportar a archivo .tar
docker save fraud-detection-api:latest -o fraud-detection-api.tar

# Verificar el archivo (debería ser ~600MB)
ls -lh fraud-detection-api.tar  # Linux/macOS
dir fraud-detection-api.tar     # Windows
```

#### Comprimir para reducir tamaño (Opcional)

```powershell
# Windows (PowerShell 5+)
Compress-Archive -Path fraud-detection-api.tar -DestinationPath fraud-detection-api.zip

# Linux/macOS
gzip fraud-detection-api.tar
# Resultado: fraud-detection-api.tar.gz (~300-400MB)
```

### Importar imagen en otro equipo

#### Descomprimir (si está comprimido)

```powershell
# Windows
Expand-Archive -Path fraud-detection-api.zip -DestinationPath .

# Linux/macOS
gunzip fraud-detection-api.tar.gz
```

#### Cargar imagen

```powershell
# Importar imagen
docker load -i fraud-detection-api.tar

# Verificar
docker images | grep fraud-detection-api
```

#### Ejecutar

```powershell
# Método 1: Docker run
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest

# Método 2: Con docker-compose (si tienes el archivo)
docker-compose up -d
```

---

## 🌍 Distribución a Otros Equipos

### Opción 1: Archivo .tar (Sin Internet)

**Ventajas:**
- ✅ No requiere conexión a internet
- ✅ Control total sobre la distribución
- ✅ Funciona en redes privadas

**Pasos:**

1. **En tu equipo (origen):**
   ```powershell
   # Exportar
   docker save fraud-detection-api:latest -o fraud-detection-api.tar
   
   # Comprimir
   gzip fraud-detection-api.tar  # O usar zip en Windows
   ```

2. **Transferir archivo:**
   - USB/Disco externo
   - Red local (carpeta compartida)
   - Email (si el tamaño lo permite)
   - Cloud storage (Google Drive, OneDrive, etc.)

3. **En el equipo destino:**
   ```powershell
   # Descomprimir
   gunzip fraud-detection-api.tar.gz  # O unzip en Windows
   
   # Importar
   docker load -i fraud-detection-api.tar
   
   # Ejecutar
   docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest
   ```

### Opción 2: Docker Hub (Con Internet)

**Ventajas:**
- ✅ Fácil de compartir
- ✅ Versionado automático
- ✅ Pull desde cualquier lugar

**Pasos:**

1. **Crear cuenta en Docker Hub:**
   - Ir a: https://hub.docker.com
   - Registrarse gratuitamente

2. **Login desde terminal:**
   ```powershell
   docker login
   # Ingresar username y password
   ```

3. **Etiquetar imagen:**
   ```powershell
   # Formato: docker tag <imagen-local> <username>/<repo>:<tag>
   docker tag fraud-detection-api:latest tuusuario/fraud-detection-api:latest
   docker tag fraud-detection-api:latest tuusuario/fraud-detection-api:1.0.0
   ```

4. **Subir a Docker Hub:**
   ```powershell
   docker push tuusuario/fraud-detection-api:latest
   docker push tuusuario/fraud-detection-api:1.0.0
   ```

5. **En otros equipos:**
   ```powershell
   # Descargar y ejecutar
   docker pull tuusuario/fraud-detection-api:latest
   docker run -d -p 8000:8000 --name fraud-api tuusuario/fraud-detection-api:latest
   ```

### Opción 3: Registro Privado (Empresarial)

Si trabajas en una empresa, puedes usar:

- **Azure Container Registry**
- **AWS ECR**
- **Google Container Registry**
- **Harbor** (self-hosted)

---

## 📋 Guía Rápida para Nuevos Equipos

Crea un archivo `INSTRUCCIONES_DOCKER.md` para distribuir:

```markdown
# Instrucciones para Ejecutar el Proyecto

## Requisitos
- Docker Desktop instalado
- Puerto 8000 disponible

## Pasos

### 1. Importar la imagen

```powershell
docker load -i fraud-detection-api.tar
```

### 2. Ejecutar

```powershell
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest
```

### 3. Verificar

Abrir navegador en: http://localhost:8000/docs

### 4. Detener

```powershell
docker stop fraud-api
```

### 5. Reiniciar

```powershell
docker start fraud-api
```

## Troubleshooting

**Error: puerto 8000 en uso**
```powershell
docker run -d -p 8080:8000 --name fraud-api fraud-detection-api:latest
# Usar http://localhost:8080 en lugar de 8000
```

**Ver logs**
```powershell
docker logs fraud-api
```
```

---

## 🐳 Docker Hub (Opcional)

### Publicación paso a paso

1. **Preparar repositorio:**
   ```powershell
   # Login
   docker login
   
   # Tag con tu username
   docker tag fraud-detection-api:latest tuusuario/fraud-detection-api:latest
   ```

2. **Push:**
   ```powershell
   docker push tuusuario/fraud-detection-api:latest
   ```

3. **Compartir:**
   - URL: `https://hub.docker.com/r/tuusuario/fraud-detection-api`
   - Comando para otros: `docker pull tuusuario/fraud-detection-api:latest`

### Versionado

```powershell
# Multiple tags para la misma imagen
docker tag fraud-detection-api:latest tuusuario/fraud-detection-api:1.0.0
docker tag fraud-detection-api:latest tuusuario/fraud-detection-api:stable

# Push todas las versiones
docker push tuusuario/fraud-detection-api:latest
docker push tuusuario/fraud-detection-api:1.0.0
docker push tuusuario/fraud-detection-api:stable
```

---

## 🛠️ Troubleshooting

### Error: "Cannot connect to Docker daemon"

**Windows:**
```powershell
# Iniciar Docker Desktop manualmente
# O reiniciar el servicio
Restart-Service docker
```

**Linux:**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### Error: "Port 8000 already in use"

```powershell
# Ver qué está usando el puerto
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/macOS

# Usar otro puerto
docker run -d -p 8080:8000 --name fraud-api fraud-detection-api:latest
```

### Imagen muy grande

```powershell
# Ver capas de la imagen
docker history fraud-detection-api:latest

# Limpiar imágenes no usadas
docker system prune -a

# Comprimir antes de transferir
gzip fraud-detection-api.tar
```

### Error: "No space left on device"

```powershell
# Limpiar contenedores detenidos
docker container prune

# Limpiar imágenes sin usar
docker image prune -a

# Limpiar todo
docker system prune -a --volumes
```

### Error al cargar modelo dentro del contenedor

```powershell
# Verificar que los archivos existen en la imagen
docker run --rm fraud-detection-api:latest ls -la models/
docker run --rm fraud-detection-api:latest ls -la data/processed/

# Reconstruir asegurando que los archivos estén
docker build --no-cache -t fraud-detection-api:latest .
```

### Logs del contenedor

```powershell
# Ver logs en tiempo real
docker logs -f fraud-api

# Últimas 100 líneas
docker logs --tail 100 fraud-api

# Desde una hora atrás
docker logs --since 1h fraud-api
```

### Entrar al contenedor para debugging

```powershell
# Bash interactivo
docker exec -it fraud-api bash

# O si bash no está disponible
docker exec -it fraud-api sh

# Ejecutar comando específico
docker exec fraud-api ls -la /app/models/
docker exec fraud-api python -c "import sys; print(sys.version)"
```

---

## 📊 Comparación de Métodos de Distribución

| Método | Tamaño | Velocidad | Internet | Privacidad | Dificultad |
|--------|--------|-----------|----------|------------|------------|
| **.tar file** | ~600MB | Media | No | ✅ Alta | Fácil |
| **.tar.gz** | ~350MB | Rápida | No | ✅ Alta | Fácil |
| **Docker Hub** | - | Muy rápida | Sí | ⚠️ Pública* | Muy fácil |
| **Registry privado** | - | Muy rápida | Sí | ✅ Alta | Media |

*Docker Hub tiene repositorios privados limitados en plan gratuito

---

## ✅ Checklist de Distribución

### Para quien exporta:

- [ ] Modelo entrenado y validado
- [ ] Imagen Docker construida correctamente
- [ ] Imagen probada localmente
- [ ] Imagen exportada a .tar
- [ ] Archivo comprimido (opcional)
- [ ] Instrucciones claras incluidas
- [ ] Documentación de endpoints

### Para quien importa:

- [ ] Docker Desktop instalado
- [ ] Imagen importada con `docker load`
- [ ] Puerto 8000 disponible
- [ ] Contenedor ejecutándose
- [ ] Health check respondiendo
- [ ] Endpoints funcionando
- [ ] Documentación revisada

---

## 🎓 Ejemplos Completos

### Ejemplo 1: Compartir en USB

**Equipo A (preparar):**
```powershell
# 1. Exportar
docker save fraud-detection-api:latest -o fraud-detection-api.tar

# 2. Comprimir
Compress-Archive -Path fraud-detection-api.tar -DestinationPath fraud-api.zip

# 3. Copiar fraud-api.zip a USB
```

**Equipo B (recibir):**
```powershell
# 1. Copiar desde USB al equipo

# 2. Descomprimir
Expand-Archive -Path fraud-api.zip -DestinationPath .

# 3. Cargar imagen
docker load -i fraud-detection-api.tar

# 4. Ejecutar
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest

# 5. Probar
start http://localhost:8000/docs
```

### Ejemplo 2: Compartir vía Docker Hub

**Equipo A (publicar):**
```powershell
docker login
docker tag fraud-detection-api:latest miusuario/fraud-api:latest
docker push miusuario/fraud-api:latest
```

**Equipo B (descargar):**
```powershell
docker pull miusuario/fraud-api:latest
docker run -d -p 8000:8000 --name fraud-api miusuario/fraud-api:latest
```

### Ejemplo 3: Red local (carpeta compartida)

**Equipo A:**
```powershell
# Exportar a carpeta de red
docker save fraud-detection-api:latest -o \\servidor\compartida\fraud-api.tar
```

**Equipo B:**
```powershell
# Cargar desde carpeta de red
docker load -i \\servidor\compartida\fraud-api.tar
docker run -d -p 8000:8000 --name fraud-api fraud-detection-api:latest
```

---

## 🔗 Referencias

- **Docker Documentation:** https://docs.docker.com/
- **Docker Hub:** https://hub.docker.com/
- **FastAPI in Docker:** https://fastapi.tiangolo.com/deployment/docker/
- **Docker Best Practices:** https://docs.docker.com/develop/dev-best-practices/

---

## 📞 Soporte

Para problemas específicos:

1. **Verificar logs:** `docker logs fraud-api`
2. **Revisar health:** http://localhost:8000/health
3. **Consultar documentación:** `docs/` en el proyecto
4. **Issues del proyecto:** GitHub Issues (si aplica)

---

**Última actualización:** Noviembre 2024  
**Versión:** 1.0.0  
**Proyecto:** MLOps Fraud Detection API
