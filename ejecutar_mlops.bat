@echo off
REM =====================================================
REM Script para ejecutar todo el pipeline MLOps
REM Uso: ejecutar_mlops.bat
REM =====================================================

echo.
echo ========================================================================
echo   PIPELINE MLOPS - DETECCION DE FRAUDE
echo ========================================================================
echo.

REM Verificar si el ambiente virtual está activado
if not defined VIRTUAL_ENV (
    echo Activando ambiente virtual...
    if exist MLOPS_FINAL-venv\Scripts\activate.bat (
        call MLOPS_FINAL-venv\Scripts\activate.bat
    ) else (
        echo ERROR: No se encontro el ambiente virtual MLOPS_FINAL-venv
        echo Por favor ejecuta primero: set_up.bat
        pause
        exit /b 1
    )
)

echo Ambiente virtual activado: %VIRTUAL_ENV%
echo.

REM Ejecutar el pipeline completo
python run_mlops.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================================
    echo   PIPELINE COMPLETADO EXITOSAMENTE
    echo ========================================================================
    echo.
    echo Quieres abrir el dashboard de monitoreo? (S/N)
    set /p OPEN_DASHBOARD="> "
    
    if /i "!OPEN_DASHBOARD!"=="S" (
        echo.
        echo Abriendo dashboard...
        streamlit run app_monitoring.py
    ) else (
        echo.
        echo Para ver el dashboard mas tarde, ejecuta:
        echo   streamlit run app_monitoring.py
    )
) else (
    echo.
    echo ========================================================================
    echo   ERROR EN EL PIPELINE
    echo ========================================================================
    pause
    exit /b 1
)

echo.
pause
