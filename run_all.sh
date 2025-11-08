#!/bin/bash
################################################################################
# SCRIPT DE EJECUCIÓN UNIVERSAL - MLOps Fraud Detection Project
# 
# Este script ejecuta todo el pipeline de MLOps en un solo comando.
# Compatible con Windows (Git Bash), Linux y macOS.
#
# Uso:
#   ./run_all.sh              # Ejecuta todo el pipeline
#   ./run_all.sh --docker     # Ejecuta usando Docker
#   ./run_all.sh --api-only   # Solo inicia la API
#   ./run_all.sh --help       # Muestra ayuda
################################################################################

set -e  # Salir si hay error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funciones de utilidad
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Función para verificar Python
check_python() {
    print_header "Verificando Python"
    
    if command -v python &> /dev/null; then
        PYTHON_CMD=python
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    else
        print_error "Python no encontrado. Instala Python 3.11+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python encontrado: $PYTHON_VERSION"
    print_success "Comando: $PYTHON_CMD"
}

# Función para crear/activar entorno virtual
setup_venv() {
    print_header "Configurando Entorno Virtual"
    
    if [ ! -d "MLOPS_FINAL-venv" ]; then
        print_info "Creando entorno virtual..."
        $PYTHON_CMD -m venv MLOPS_FINAL-venv
        print_success "Entorno virtual creado"
    else
        print_success "Entorno virtual ya existe"
    fi
    
    # Activar entorno virtual según el sistema operativo
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows (Git Bash)
        source MLOPS_FINAL-venv/Scripts/activate
        print_success "Entorno virtual activado (Windows)"
    else
        # Linux/macOS
        source MLOPS_FINAL-venv/bin/activate
        print_success "Entorno virtual activado (Unix)"
    fi
}

# Función para instalar dependencias
install_dependencies() {
    print_header "Instalando Dependencias"
    
    print_info "Actualizando pip..."
    $PYTHON_CMD -m pip install --upgrade pip setuptools wheel --quiet
    
    print_info "Instalando dependencias del proyecto..."
    pip install -r requirements.txt --quiet
    
    print_success "Dependencias instaladas correctamente"
}

# Función para ejecutar el pipeline completo
run_pipeline() {
    print_header "Ejecutando Pipeline MLOps Completo"
    
    print_info "Paso 1/4: Feature Engineering..."
    $PYTHON_CMD mlops_pipeline/src/ft_engineering.py
    print_success "Feature Engineering completado"
    
    print_info "Paso 2/4: Entrenamiento de Modelos..."
    $PYTHON_CMD mlops_pipeline/src/model_training_evaluation.py
    print_success "Entrenamiento completado"
    
    print_info "Paso 3/4: Monitoreo de Datos..."
    $PYTHON_CMD mlops_pipeline/src/model_monitoring.py
    print_success "Monitoreo completado"
    
    print_info "Paso 4/4: Dashboard de Monitoreo..."
    print_info "Iniciando Streamlit en segundo plano..."
    
    # Iniciar Streamlit en segundo plano
    nohup streamlit run app_monitoring.py --server.port 8501 > streamlit.log 2>&1 &
    STREAMLIT_PID=$!
    
    echo $STREAMLIT_PID > streamlit.pid
    print_success "Dashboard iniciado en http://localhost:8501 (PID: $STREAMLIT_PID)"
}

# Función para solo iniciar la API
run_api_only() {
    print_header "Iniciando API de Detección de Fraude"
    
    print_info "Verificando modelo entrenado..."
    if [ ! -f "models/best_model.pkl" ]; then
        print_error "Modelo no encontrado. Ejecuta el pipeline completo primero."
        exit 1
    fi
    
    print_success "Modelo encontrado"
    
    print_info "Instalando dependencias de la API..."
    pip install -r api/requirements.txt --quiet
    
    print_info "Iniciando API en http://localhost:8000..."
    $PYTHON_CMD -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
}

# Función para ejecutar con Docker
run_docker() {
    print_header "Ejecutando con Docker"
    
    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker no encontrado. Instala Docker Desktop."
        exit 1
    fi
    
    print_success "Docker encontrado"
    
    # Verificar si el modelo existe
    if [ ! -f "models/best_model.pkl" ]; then
        print_info "Modelo no encontrado. Ejecutando pipeline primero..."
        setup_venv
        install_dependencies
        run_pipeline
        
        # Esperar a que termine
        print_info "Esperando a que termine el pipeline..."
        sleep 5
    fi
    
    print_info "Construyendo imagen Docker..."
    docker build -t fraud-detection-api:latest .
    print_success "Imagen construida"
    
    print_info "Iniciando contenedor..."
    docker-compose up -d
    print_success "Contenedor iniciado"
    
    echo ""
    print_success "API disponible en: http://localhost:8000"
    print_success "Documentación: http://localhost:8000/docs"
    print_success "Health Check: http://localhost:8000/health"
    echo ""
    print_info "Para ver logs: docker-compose logs -f"
    print_info "Para detener: docker-compose down"
}

# Función para mostrar ayuda
show_help() {
    cat << EOF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MLOps Fraud Detection - Script de Ejecución Universal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USO:
  ./run_all.sh [OPCIÓN]

OPCIONES:
  (sin opciones)    Ejecuta el pipeline completo de MLOps
  --docker          Ejecuta todo usando Docker (recomendado para portabilidad)
  --api-only        Solo inicia la API (requiere modelo pre-entrenado)
  --help            Muestra esta ayuda

EJEMPLOS:
  ./run_all.sh                 # Pipeline completo local
  ./run_all.sh --docker        # Todo en Docker
  ./run_all.sh --api-only      # Solo API

COMPONENTES DEL PIPELINE:
  1. Feature Engineering       → Preprocesamiento de datos
  2. Model Training            → Entrenamiento de múltiples modelos
  3. Data Monitoring           → Monitoreo de drift y alertas
  4. Dashboard (Streamlit)     → Visualización en http://localhost:8501
  5. API (FastAPI)             → Predicciones en http://localhost:8000

SALIDAS:
  • Dashboard: http://localhost:8501
  • API Docs: http://localhost:8000/docs
  • Logs: streamlit.log
  • Resultados: outputs/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
}

# Función para limpiar procesos anteriores
cleanup() {
    print_header "Limpiando Procesos Anteriores"
    
    if [ -f "streamlit.pid" ]; then
        STREAMLIT_PID=$(cat streamlit.pid)
        if ps -p $STREAMLIT_PID > /dev/null 2>&1; then
            print_info "Deteniendo Streamlit anterior (PID: $STREAMLIT_PID)..."
            kill $STREAMLIT_PID 2>/dev/null || true
            rm streamlit.pid
            print_success "Streamlit detenido"
        else
            rm streamlit.pid
        fi
    fi
}

# Función principal
main() {
    clear
    
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      MLOps - Sistema de Detección de Fraude                 ║
║      Ejecución Universal del Pipeline Completo              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
    
    # Parsear argumentos
    case "${1:-}" in
        --docker)
            run_docker
            ;;
        --api-only)
            check_python
            setup_venv
            run_api_only
            ;;
        --help|-h)
            show_help
            ;;
        "")
            cleanup
            check_python
            setup_venv
            install_dependencies
            run_pipeline
            
            echo ""
            print_header "Pipeline Completado Exitosamente"
            echo ""
            print_success "Dashboard: http://localhost:8501"
            print_success "Logs: streamlit.log"
            print_success "Resultados: outputs/"
            echo ""
            print_info "Para iniciar la API: ./run_all.sh --api-only"
            print_info "Para usar Docker: ./run_all.sh --docker"
            echo ""
            ;;
        *)
            print_error "Opción no reconocida: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Trap para manejar Ctrl+C
trap 'echo -e "\n${RED}Interrumpido por usuario${NC}"; exit 130' INT

# Ejecutar
main "$@"
