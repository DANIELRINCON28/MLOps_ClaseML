@echo off
REM ===================================
REM Script para preparar el ambiente virtual de Python e instalar las librerías necesarias
REM ===================================
REM NO DEBES MODIFICAR ESTE ARCHIVO
REM ===================================
REM Purpose: Script to setup a Python virtual environment, install requirements
REM ===================================

setlocal EnableDelayedExpansion

echo.
echo ========================================================================
echo   CONFIGURACION DEL AMBIENTE VIRTUAL - MLOps
echo ========================================================================
echo.

REM Verificar que Python esté instalado
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python no esta instalado o no esta en el PATH del sistema.
    echo Por favor instala Python 3.8 o superior desde https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python encontrado:
python --version
echo.

REM Desactivar el ambiente virtual actual si está activo
if defined VIRTUAL_ENV (
    echo [*] Desactivando ambiente virtual actual...
    call deactivate 2>nul
)

echo [*] Leyendo configuracion del proyecto desde config.json...

REM Leer línea que contiene "project_code"
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

echo [OK] Codigo del proyecto: !project_code!
echo.

REM Verificar si el ambiente virtual ya existe
if exist !project_code!-venv (
    echo [ADVERTENCIA] El ambiente virtual !project_code!-venv ya existe.
    echo Deseas eliminarlo y crear uno nuevo? (S/N)
    set /p RECREATE="> "
    if /i "!RECREATE!"=="S" (
        echo [*] Eliminando ambiente virtual existente...
        rmdir /s /q !project_code!-venv
        echo [OK] Ambiente virtual eliminado.
    ) else (
        echo [*] Usando ambiente virtual existente...
        goto :activate_env
    )
)

echo [*] Creando nuevo ambiente virtual: !project_code!-venv
python -m venv !project_code!-venv

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error al crear el ambiente virtual.
    pause
    exit /b 1
)

echo [OK] Ambiente virtual creado exitosamente.
echo.

:activate_env
echo [*] Activando ambiente virtual...
call !project_code!-venv\Scripts\activate.bat

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error al activar el ambiente virtual.
    pause
    exit /b 1
)

echo [OK] Ambiente virtual activado: %VIRTUAL_ENV%
echo.

REM Actualizar pip
echo [*] Actualizando pip...
python -m pip install --upgrade pip --quiet

echo.
echo ========================================================================
echo   INSTALANDO DEPENDENCIAS
echo ========================================================================
echo.

if not exist requirements.txt (
    echo [ERROR] requirements.txt no fue encontrado en el directorio actual.
    pause
    exit /b 1
)

echo [*] Instalando librerias desde requirements.txt...
echo     (Esto puede tomar varios minutos...)
echo.

pip install --no-cache-dir -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Error instalando las librerias desde requirements.txt.
    echo Por favor revisa los mensajes de error anteriores.
    pause
    exit /b 1
)

echo.
echo [OK] Todas las librerias instaladas correctamente.
echo.

echo ========================================================================
echo   REGISTRANDO KERNEL DE JUPYTER
echo ========================================================================
echo.

echo [*] Registrando kernel con Jupyter...
python -m ipykernel install --user --name=!project_code!-venv --display-name="!project_code!-venv Python ETL"

if %ERRORLEVEL% EQU 0 (
    echo [OK] Kernel de Jupyter registrado correctamente.
    echo     Nombre: "!project_code!-venv Python ETL"
) else (
    echo [ADVERTENCIA] No se pudo registrar el kernel de Jupyter.
    echo              Los notebooks pueden no reconocer este ambiente.
)

echo.
echo ========================================================================
echo   CREANDO ESTRUCTURA DE DIRECTORIOS
echo ========================================================================
echo.

REM Crear directorios necesarios si no existen
if not exist "data\processed" (
    echo [*] Creando directorio: data\processed
    mkdir "data\processed"
)

if not exist "models" (
    echo [*] Creando directorio: models
    mkdir "models"
)

if not exist "outputs\monitoring" (
    echo [*] Creando directorio: outputs\monitoring
    mkdir "outputs\monitoring"
)

echo [OK] Estructura de directorios verificada.
echo.

echo ========================================================================
echo   CONFIGURACION COMPLETADA EXITOSAMENTE
echo ========================================================================
echo.
echo El ambiente virtual esta listo para usar.
echo.
echo PROXIMOS PASOS:
echo   1. Para ejecutar el proyecto completo, ejecuta:
echo      ejecutar_mlops.bat
echo.
echo   2. O activa el ambiente manualmente y ejecuta:
echo      !project_code!-venv\Scripts\activate
echo      python run_mlops.py
echo.
echo ========================================================================
echo.
pause

