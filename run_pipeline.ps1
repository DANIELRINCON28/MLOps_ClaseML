# ============================================================
# PIPELINE MLOPS COMPLETO - EJECUCION AUTOMATIZADA
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   PIPELINE MLOPS - DETECCION DE FRAUDE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar entorno virtual
Write-Host "[1/5] Verificando entorno virtual..." -ForegroundColor Yellow

if (-not (Test-Path "MLOPS_FINAL-venv\Scripts\python.exe")) {
    Write-Host "   ERROR: Entorno virtual no encontrado. Ejecutando set_up.bat..." -ForegroundColor Red
    cmd /c "set_up.bat"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Fallo la configuracion del entorno" -ForegroundColor Red
        pause
        exit 1
    }
} else {
    Write-Host "   OK: Entorno virtual encontrado" -ForegroundColor Green
}

# 2. Activar entorno virtual
Write-Host ""
Write-Host "[2/5] Activando entorno virtual..." -ForegroundColor Yellow
& ".\MLOPS_FINAL-venv\Scripts\Activate.ps1"
Write-Host "   OK: Entorno activado" -ForegroundColor Green

# 3. Verificar e instalar dependencias críticas
Write-Host ""
Write-Host "[3/5] Instalando dependencias criticas (si falta alguna)..." -ForegroundColor Yellow
pip install -q pandas scikit-learn xgboost lightgbm imbalanced-learn streamlit matplotlib seaborn plotly joblib
Write-Host "   OK: Dependencias verificadas" -ForegroundColor Green

# 4. Ejecutar Feature Engineering
Write-Host ""
Write-Host "[4/5] Ejecutando Feature Engineering..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
python mlops_pipeline/src/ft_engineering.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR en Feature Engineering" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "   OK: Feature Engineering completado" -ForegroundColor Green

# 5. Ejecutar Model Training
Write-Host ""
Write-Host "[5/5] Ejecutando Model Training and Evaluation..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "NOTA: Este proceso puede tomar varios minutos..." -ForegroundColor Yellow
Write-Host "      Entrenando 5 modelos con ~200k muestras" -ForegroundColor Yellow
Write-Host ""

python mlops_pipeline/src/model_training_evaluation.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR en Model Training" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "   OK: Model Training completado" -ForegroundColor Green

# 6. Lanzar Dashboard Streamlit
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   LANZANDO DASHBOARD DE MONITOREO" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Iniciando Streamlit Dashboard..." -ForegroundColor Green
Write-Host "   URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "   Presiona Ctrl+C para detener el servidor" -ForegroundColor Yellow
Write-Host ""

streamlit run mlops_pipeline/src/app_monitoring.py
