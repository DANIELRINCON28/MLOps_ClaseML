@echo off
REM =====================================================
REM Script para ejecutar todo el pipeline MLOps
REM Uso: ejecutar_mlops.bat
REM =====================================================

setlocal EnableDelayedExpansion

echo.
echo ========================================================================
echo   PIPELINE MLOPS - DETECCION DE FRAUDE
echo ========================================================================
echo.

REM Leer el código del proyecto desde config.json
for /f "usebackq tokens=2 delims=:" %%A in (`findstr "project_code" config.json`) do (
    set "line=%%A"
    set "line=!line:,=!"
    set "line=!line:"=!"
    set "project_code=!line:~1!"
)

if "!project_code!"=="" (
    echo [ERROR] No se pudo leer el project_code desde config.json
    pause
    exit /b 1
)

REM Verificar si el ambiente virtual está activado
if not defined VIRTUAL_ENV (
    echo [*] Activando ambiente virtual: !project_code!-venv
    if exist !project_code!-venv\Scripts\activate.bat (
        call !project_code!-venv\Scripts\activate.bat
    ) else (
        echo.
        echo [ERROR] No se encontro el ambiente virtual: !project_code!-venv
        echo.
        echo Por favor ejecuta primero: set_up.bat
        echo.
        pause
        exit /b 1
    )
)

echo [OK] Ambiente virtual activado: %VIRTUAL_ENV%
echo.

REM Verificar que Python funcione correctamente
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python no esta disponible en el ambiente virtual.
    pause
    exit /b 1
)

echo ========================================================================
echo   EJECUTANDO PIPELINE COMPLETO
echo ========================================================================
echo.
echo [*] Iniciando pipeline de MLOps...
echo     - Feature Engineering
echo     - Entrenamiento de Modelos
echo     - Evaluacion y Comparacion
echo     - Monitoreo de Datos
echo.

REM Ejecutar el pipeline completo
python run_mlops.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================================
    echo   PIPELINE COMPLETADO EXITOSAMENTE
    echo ========================================================================
    echo.
    echo Resultados guardados en:
    echo   - Modelos: models/
    echo   - Metricas: outputs/
    echo   - Monitoreo: outputs/monitoring/
    echo.
    echo ========================================================================
    echo   ABRIENDO DASHBOARD INTERACTIVO
    echo ========================================================================
    echo.
    echo [*] Iniciando dashboard en http://localhost:8501...
    echo     Presiona Ctrl+C para detener el servidor
    echo.
    streamlit run app_monitoring.py
) else (
    echo.
    echo ========================================================================
    echo   ERROR EN EL PIPELINE
    echo ========================================================================
    echo.
    echo Revisa los mensajes de error anteriores para mas detalles.
    echo.
    pause
    exit /b 1
)

echo.
pause
