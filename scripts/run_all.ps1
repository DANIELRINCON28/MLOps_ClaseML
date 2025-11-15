#Requires -Version 5.1
<#
.SYNOPSIS
    Script de ejecución universal para el proyecto MLOps de Detección de Fraude

.DESCRIPTION
    Ejecuta todo el pipeline de MLOps en un solo comando.
    Incluye Feature Engineering, Model Training, Monitoring y Dashboard.

.PARAMETER Docker
    Ejecuta usando Docker en lugar de instalación local

.PARAMETER ApiOnly
    Solo inicia la API (requiere modelo pre-entrenado)

.PARAMETER Help
    Muestra ayuda detallada

.EXAMPLE
    .\run_all.ps1
    Ejecuta el pipeline completo

.EXAMPLE
    .\run_all.ps1 -Docker
    Ejecuta todo usando Docker

.EXAMPLE
    .\run_all.ps1 -ApiOnly
    Solo inicia la API
#>

[CmdletBinding()]
param(
    [switch]$Docker,
    [switch]$ApiOnly,
    [switch]$Help
)

# Configuración de colores
$Host.UI.RawUI.WindowTitle = "MLOps - Fraud Detection Pipeline"

function Write-Header {
    param([string]$Message)
    
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Blue
    Write-Host "  $Message" -ForegroundColor Blue
    Write-Host "================================================================" -ForegroundColor Blue
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "[*] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Show-Banner {
    Clear-Host
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "                                                                " -ForegroundColor Cyan
    Write-Host "      MLOps - Sistema de Deteccion de Fraude                   " -ForegroundColor Cyan
    Write-Host "      Ejecucion Universal del Pipeline Completo                " -ForegroundColor Cyan
    Write-Host "                                                                " -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Write-Host "================================================================"
    Write-Host "  MLOps Fraud Detection - Script de Ejecucion Universal"
    Write-Host "================================================================"
    Write-Host ""
    Write-Host "USO:"
    Write-Host "  .\run_all.ps1 [OPCION]"
    Write-Host ""
    Write-Host "OPCIONES:"
    Write-Host "  (sin opciones)    Ejecuta el pipeline completo de MLOps"
    Write-Host "  -Docker           Ejecuta todo usando Docker (recomendado para portabilidad)"
    Write-Host "  -ApiOnly          Solo inicia la API (requiere modelo pre-entrenado)"
    Write-Host "  -Help             Muestra esta ayuda"
    Write-Host ""
    Write-Host "EJEMPLOS:"
    Write-Host "  .\run_all.ps1                 # Pipeline completo local"
    Write-Host "  .\run_all.ps1 -Docker         # Todo en Docker"
    Write-Host "  .\run_all.ps1 -ApiOnly        # Solo API"
    Write-Host ""
    Write-Host "COMPONENTES DEL PIPELINE:"
    Write-Host "  1. Feature Engineering       - Preprocesamiento de datos"
    Write-Host "  2. Model Training            - Entrenamiento de multiples modelos"
    Write-Host "  3. Data Monitoring           - Monitoreo de drift y alertas"
    Write-Host "  4. Dashboard (Streamlit)     - Visualizacion en http://localhost:8501"
    Write-Host "  5. API (FastAPI)             - Predicciones en http://localhost:8000"
    Write-Host ""
    Write-Host "SALIDAS:"
    Write-Host "  - Dashboard: http://localhost:8501"
    Write-Host "  - API Docs: http://localhost:8000/docs"
    Write-Host "  - Logs: streamlit.log"
    Write-Host "  - Resultados: outputs/"
    Write-Host ""
    Write-Host "================================================================"
}

function Test-Python {
    Write-Header "Verificando Python"
    
    $pythonCmd = $null
    
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonCmd = "python"
    }
    elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $pythonCmd = "python3"
    }
    else {
        Write-Error "Python no encontrado. Instala Python 3.11+"
        exit 1
    }
    
    $version = & $pythonCmd --version 2>&1
    Write-Success "Python encontrado: $version"
    Write-Success "Comando: $pythonCmd"
    
    return $pythonCmd
}

function Setup-VirtualEnv {
    param([string]$PythonCmd)
    
    Write-Header "Configurando Entorno Virtual"
    
    $venvPath = "MLOPS_FINAL-venv"
    
    if (-not (Test-Path $venvPath)) {
        Write-Info "Creando entorno virtual..."
        & $PythonCmd -m venv $venvPath
        Write-Success "Entorno virtual creado"
    }
    else {
        Write-Success "Entorno virtual ya existe"
    }
    
    # Activar entorno virtual
    Write-Info "Activando entorno virtual..."
    & "$venvPath\Scripts\Activate.ps1"
    Write-Success "Entorno virtual activado"
}

function Install-Dependencies {
    param([string]$PythonCmd)
    
    Write-Header "Instalando Dependencias"
    
    Write-Info "Actualizando pip..."
    & $PythonCmd -m pip install --upgrade pip setuptools wheel --quiet
    
    Write-Info "Instalando dependencias del proyecto..."
    & pip install -r requirements.txt --quiet
    
    Write-Success "Dependencias instaladas correctamente"
}

function Start-Pipeline {
    param([string]$PythonCmd)
    
    Write-Header "Ejecutando Pipeline MLOps Completo"
    
    Write-Info "Paso 1/4: Feature Engineering..."
    & $PythonCmd mlops_pipeline/src/ft_engineering.py
    Write-Success "Feature Engineering completado"
    
    Write-Info "Paso 2/4: Entrenamiento de Modelos..."
    & $PythonCmd mlops_pipeline/src/model_training_evaluation.py
    Write-Success "Entrenamiento completado"
    
    Write-Info "Paso 3/4: Monitoreo de Datos..."
    & $PythonCmd mlops_pipeline/src/model_monitoring.py
    Write-Success "Monitoreo completado"
    
    Write-Info "Paso 4/4: Dashboard de Monitoreo..."
    Write-Info "Iniciando Streamlit en segundo plano..."
    
    # Iniciar Streamlit en segundo plano
    $streamlitProcess = Start-Process streamlit -ArgumentList "run", "app_monitoring.py", "--server.port", "8501" -PassThru -WindowStyle Hidden
    
    $streamlitProcess.Id | Out-File -FilePath "streamlit.pid"
    Write-Success "Dashboard iniciado en http://localhost:8501 (PID: $($streamlitProcess.Id))"
}

function Start-ApiOnly {
    param([string]$PythonCmd)
    
    Write-Header "Iniciando API de Detección de Fraude"
    
    Write-Info "Verificando modelo entrenado..."
    if (-not (Test-Path "models/best_model.pkl")) {
        Write-Error "Modelo no encontrado. Ejecuta el pipeline completo primero."
        exit 1
    }
    
    Write-Success "Modelo encontrado"
    
    Write-Info "Instalando dependencias de la API..."
    & pip install -r api/requirements.txt --quiet
    
    Write-Info "Iniciando API en http://localhost:8000..."
    & $PythonCmd -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
}

function Start-Docker {
    Write-Header "Ejecutando con Docker"
    
    # Verificar Docker
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker no encontrado. Instala Docker Desktop."
        exit 1
    }
    
    Write-Success "Docker encontrado"
    
    # Verificar si el modelo existe
    if (-not (Test-Path "models/best_model.pkl")) {
        Write-Info "Modelo no encontrado. Ejecutando pipeline primero..."
        
        $pythonCmd = Test-Python
        Setup-VirtualEnv -PythonCmd $pythonCmd
        Install-Dependencies -PythonCmd $pythonCmd
        Start-Pipeline -PythonCmd $pythonCmd
        
        Write-Info "Esperando a que termine el pipeline..."
        Start-Sleep -Seconds 5
    }
    
    Write-Info "Construyendo imagen Docker..."
    docker build -t fraud-detection-api:latest .
    Write-Success "Imagen construida"
    
    Write-Info "Iniciando contenedor..."
    docker-compose up -d
    Write-Success "Contenedor iniciado"
    
    Write-Host ""
    Write-Success "API disponible en: http://localhost:8000"
    Write-Success "Documentación: http://localhost:8000/docs"
    Write-Success "Health Check: http://localhost:8000/health"
    Write-Host ""
    Write-Info "Para ver logs: docker-compose logs -f"
    Write-Info "Para detener: docker-compose down"
}

function Stop-PreviousProcesses {
    Write-Header "Limpiando Procesos Anteriores"
    
    if (Test-Path "streamlit.pid") {
        $streamlitPid = Get-Content "streamlit.pid"
        
        if (Get-Process -Id $streamlitPid -ErrorAction SilentlyContinue) {
            Write-Info "Deteniendo Streamlit anterior (PID: $streamlitPid)..."
            Stop-Process -Id $streamlitPid -Force -ErrorAction SilentlyContinue
            Remove-Item "streamlit.pid"
            Write-Success "Streamlit detenido"
        }
        else {
            Remove-Item "streamlit.pid"
        }
    }
}

# ============================================================================
# MAIN
# ============================================================================

Show-Banner

# Manejar argumentos
if ($Help) {
    Show-Help
    exit 0
}

if ($Docker) {
    Start-Docker
}
elseif ($ApiOnly) {
    $pythonCmd = Test-Python
    Setup-VirtualEnv -PythonCmd $pythonCmd
    Start-ApiOnly -PythonCmd $pythonCmd
}
else {
    # Pipeline completo
    Stop-PreviousProcesses
    
    $pythonCmd = Test-Python
    Setup-VirtualEnv -PythonCmd $pythonCmd
    Install-Dependencies -PythonCmd $pythonCmd
    Start-Pipeline -PythonCmd $pythonCmd
    
    Write-Host ""
    Write-Header "Pipeline Completado Exitosamente"
    Write-Host ""
    Write-Success "Dashboard: http://localhost:8501"
    Write-Success "Logs: streamlit.log"
    Write-Success "Resultados: outputs/"
    Write-Host ""
    Write-Info "Para iniciar la API: .\run_all.ps1 -ApiOnly"
    Write-Info "Para usar Docker: .\run_all.ps1 -Docker"
    Write-Host ""
}
