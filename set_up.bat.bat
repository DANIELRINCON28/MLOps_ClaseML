REM #Script para preparar el ambiente virtual de Python e instalar las librerías necesarias
REM ===================================
REM NO DEBES MODIFICAR ESTE ARCHIVO
REM ===================================
REM Purpose: Script to setup a Python virtual environment, install requirements, 
REM ===================================
REM NO DEBES MODIFICAR ESTE ARCHIVO

echo.
echo === Python Virtual Environment Setup ===
echo.

REM Desactivar el ambiente virtual actual si está activo
if defined VIRTUAL_ENV (
    echo Desactivando ambiente virtual actual: %VIRTUAL_ENV%
    call deactivate
)

echo Buscando código del proyecto en config.json...

@echo off
setlocal EnableDelayedExpansion

REM Cambiar al directorio donde está config.json
cd mlops_pipeline\src

REM Leer línea que contiene "project_name"
for /f "usebackq tokens=2 delims=:" %%A in (`findstr "project_code" config.json`) do (
    set "line=%%A"
    set "line=!line:,=!"
    set "line=!line:"=!"
    set "project_code=!line:~1!"
)

REM Volver al directorio raíz
cd ..\..

echo Creando nuevo ambiente virtual: %project_code%-venv
py -m venv %project_code%-venv

echo Activating virtual environment...
call %project_code%-venv\Scripts\activate

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Ambiente virtaul creado con exito!.
    echo Python actual: 
    where python
    
    echo.
    echo === Instalando requisitos ===
    if exist requirements.txt (
        echo requirements.txt encontrado, instalando librerias...
        pip install --no-cache-dir -r requirements.txt
        
        if %ERRORLEVEL% EQU 0 (
            echo.
            echo Todas las librerías instaladas correctamente.

            echo.
            echo === Registrando ambiente virtual con Jupyter ===
            echo Registrando kernel con Jupyter...
            python -m ipykernel install --user --name=%project_code%-venv --display-name="%project_code%-venv Python ETL"
            
            if %ERRORLEVEL% EQU 0 (
                echo Ambiente virtual registrado como kernel de Jupyter correctamente.
                echo Ahora puedes seleccionar "%project_code%-venv Python ETL" en Jupyter notebook.
            ) else (
                echo Advertencia: Fallo al registrar el ambiente virtual como kernel de Jupyter. Jupyter notebook puede no reconocer este ambiente virtual.
            )

        ) else (
            echo.
            echo Error instalando las librerías desde requirements.txt. Revisar los mensajes de error.
        )
    ) else (
        echo.
        echo Advertencias: requirements.txt no fue en contrado en el directorio actual.
    )
) else (
    echo.
    echo Error activando el ambiente virtual.
)

