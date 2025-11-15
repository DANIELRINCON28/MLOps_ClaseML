# ============================================================================
# Dockerfile para API de Detección de Fraude
# ============================================================================
# Este Dockerfile crea una imagen Docker para la API FastAPI que sirve
# el modelo de ML de detección de fraude.
#
# Construcción:
#   docker build -t fraud-detection-api .
#
# Ejecución:
#   docker run -p 8000:8000 fraud-detection-api
#
# Acceso:
#   http://localhost:8000
#   http://localhost:8000/docs (Swagger UI)
# ============================================================================

# Imagen base - Python 3.11 slim (ligera)
FROM python:3.11-slim

# Metadata de la imagen
LABEL maintainer="DANIEL ALEJANDRO RINCON VALENCIA"
LABEL description="API de detección de fraude en transacciones financieras"
LABEL version="1.0.0"
LABEL university="Universidad Católica Luis Amigó"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema (si son necesarias)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY api/requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la API
COPY api/ ./api/

# Copiar modelo y datos procesados
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Crear directorio para logs
RUN mkdir -p /app/logs

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Comando de inicio
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
