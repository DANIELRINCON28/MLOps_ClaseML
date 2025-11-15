# ===========================================================================
# Script de Gestion de Docker - Proyecto MLOps
# ===========================================================================
# Script para facilitar la gestion de los contenedores Docker del proyecto
# 
# Uso:
#   .\docker-manager.ps1 [comando]
#
# Comandos disponibles:
#   build     - Construye las imagenes Docker
#   up        - Levanta los contenedores
#   down      - Detiene y elimina los contenedores
#   restart   - Reinicia los contenedores
#   logs      - Muestra los logs de los contenedores
#   status    - Muestra el estado de los contenedores
#   clean     - Limpia imagenes y contenedores no usados
# ===========================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet('build', 'up', 'down', 'restart', 'logs', 'status', 'clean', 'help')]
    [string]$Command = 'help'
)

# Colores para output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Color
}

# Cambiar al directorio de configuración
$ConfigPath = Join-Path $PSScriptRoot "config"
Set-Location $ConfigPath

# Banner
function Show-Banner {
    Write-Host "`n==================================================================" -ForegroundColor Cyan
    Write-Host "          MLOps - Sistema de Deteccion de Fraude                 " -ForegroundColor Cyan
    Write-Host "          Universidad Catolica Luis Amigo                        " -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Cyan
}

# Funcion de ayuda
function Show-Help {
    Show-Banner
    Write-Host "`nComandos disponibles:`n" -ForegroundColor Yellow
    Write-Host "  build     - Construye las imagenes Docker" -ForegroundColor White
    Write-Host "  up        - Levanta los contenedores (API + Dashboard)" -ForegroundColor White
    Write-Host "  down      - Detiene y elimina los contenedores" -ForegroundColor White
    Write-Host "  restart   - Reinicia los contenedores" -ForegroundColor White
    Write-Host "  logs      - Muestra los logs de los contenedores" -ForegroundColor White
    Write-Host "  status    - Muestra el estado de los contenedores" -ForegroundColor White
    Write-Host "  clean     - Limpia imagenes y contenedores no usados" -ForegroundColor White
    Write-Host "  help      - Muestra esta ayuda`n" -ForegroundColor White
    
    Write-Host "Ejemplos de uso:`n" -ForegroundColor Yellow
    Write-Host "  .\docker-manager.ps1 build" -ForegroundColor Gray
    Write-Host "  .\docker-manager.ps1 up" -ForegroundColor Gray
    Write-Host "  .\docker-manager.ps1 logs`n" -ForegroundColor Gray
}

# Construir imagenes
function Build-Images {
    Show-Banner
    Write-Host "`nConstruyendo imagenes Docker...`n" -ForegroundColor Yellow
    
    Write-Host "Construyendo imagen de API..." -ForegroundColor Cyan
    docker-compose build fraud-detection-api
    
    Write-Host "`nConstruyendo imagen de Dashboard..." -ForegroundColor Cyan
    docker-compose build fraud-monitoring-dashboard
    
    Write-Host "`nImagenes construidas exitosamente!" -ForegroundColor Green
}

# Levantar contenedores
function Start-Containers {
    Show-Banner
    Write-Host "`nLevantando contenedores...`n" -ForegroundColor Yellow
    
    docker-compose up -d
    
    Write-Host "`nEsperando que los servicios esten listos..." -ForegroundColor Cyan
    Start-Sleep -Seconds 10
    
    Write-Host "`nContenedores iniciados!`n" -ForegroundColor Green
    Write-Host "Servicios disponibles:" -ForegroundColor Yellow
    Write-Host "   API FastAPI:        http://localhost:8000" -ForegroundColor White
    Write-Host "   API Docs (Swagger): http://localhost:8000/docs" -ForegroundColor White
    Write-Host "   Dashboard Streamlit: http://localhost:8501`n" -ForegroundColor White
}

# Detener contenedores
function Stop-Containers {
    Show-Banner
    Write-Host "`nDeteniendo contenedores...`n" -ForegroundColor Yellow
    
    docker-compose down
    
    Write-Host "`nContenedores detenidos!" -ForegroundColor Green
}

# Reiniciar contenedores
function Restart-Containers {
    Show-Banner
    Write-Host "`nReiniciando contenedores...`n" -ForegroundColor Yellow
    
    docker-compose restart
    
    Write-Host "`nContenedores reiniciados!" -ForegroundColor Green
}

# Ver logs
function Show-Logs {
    Show-Banner
    Write-Host "`nMostrando logs de los contenedores...`n" -ForegroundColor Yellow
    Write-Host "Presiona Ctrl+C para salir`n" -ForegroundColor Gray
    
    docker-compose logs -f
}

# Ver estado
function Show-Status {
    Show-Banner
    Write-Host "`nEstado de los contenedores:`n" -ForegroundColor Yellow
    
    docker-compose ps
    
    Write-Host "`nUso de recursos:`n" -ForegroundColor Yellow
    docker stats --no-stream fraud-api fraud-dashboard 2>$null
    
    Write-Host "`nPuertos expuestos:`n" -ForegroundColor Yellow
    docker ps --filter "name=fraud" --format "table {{.Names}}`t{{.Ports}}" 2>$null
}

# Limpiar recursos
function Clean-Resources {
    Show-Banner
    Write-Host "`nLimpiando recursos Docker...`n" -ForegroundColor Yellow
    
    $response = Read-Host "Desea limpiar imagenes y contenedores no usados? (S/N)"
    
    if ($response -eq 'S' -or $response -eq 's') {
        Write-Host "`nEliminando contenedores detenidos..." -ForegroundColor Cyan
        docker container prune -f
        
        Write-Host "Eliminando imagenes sin usar..." -ForegroundColor Cyan
        docker image prune -f
        
        Write-Host "Eliminando volumenes sin usar..." -ForegroundColor Cyan
        docker volume prune -f
        
        Write-Host "`nLimpieza completada!" -ForegroundColor Green
    } else {
        Write-Host "`nLimpieza cancelada" -ForegroundColor Yellow
    }
}

# Ejecutar comando
switch ($Command) {
    'build' { Build-Images }
    'up' { Start-Containers }
    'down' { Stop-Containers }
    'restart' { Restart-Containers }
    'logs' { Show-Logs }
    'status' { Show-Status }
    'clean' { Clean-Resources }
    'help' { Show-Help }
    default { Show-Help }
}
