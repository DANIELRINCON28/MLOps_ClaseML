@echo off
echo ============================================================
echo EJECUTANDO PIPELINE MLOPS COMPLETO
echo ============================================================
echo.

REM Activar entorno virtual
call MLOPS_FINAL-venv\Scripts\activate.bat

echo [1/3] Ejecutando Feature Engineering...
python mlops_pipeline/src/ft_engineering.py
if %errorlevel% neq 0 (
    echo ERROR en Feature Engineering
    exit /b 1
)

echo.
echo [2/3] Ejecutando Entrenamiento de Modelos...
python mlops_pipeline/src/model_training_evaluation.py
if %errorlevel% neq 0 (
    echo ERROR en Entrenamiento
    exit /b 1
)

echo.
echo [3/3] Ejecutando Monitoreo...
python mlops_pipeline/src/model_monitoring.py
if %errorlevel% neq 0 (
    echo ERROR en Monitoreo
    exit /b 1
)

echo.
echo ============================================================
echo PIPELINE COMPLETADO EXITOSAMENTE
echo ============================================================
pause
