"""
Script maestro para ejecutar todo el pipeline MLOps con un solo comando
========================================================================
Ejecuta: python run_mlops.py
O para abrir el dashboard automaticamente: python run_mlops.py --dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_command(description, command, cwd=None):
    """Ejecuta un comando y maneja errores"""
    print(f"[*] {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(f"[OK] {description} - Completado")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error en {description}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    print_header("PIPELINE MLOPS COMPLETO - DETECCION DE FRAUDE")
    
    # Directorio base
    base_dir = Path(__file__).parent
    src_dir = base_dir / "mlops_pipeline" / "src"
    
    # Verificar si los datos ya están procesados
    data_files = [
        base_dir / "data" / "processed" / "X_train.pkl",
        base_dir / "data" / "processed" / "X_test.pkl",
        base_dir / "data" / "processed" / "preprocessor.pkl"
    ]
    
    data_ready = all(f.exists() for f in data_files)
    
    # 1. Feature Engineering (solo si es necesario)
    if not data_ready:
        print_header("PASO 1: Feature Engineering")
        print("[*] Procesando datos desde cero...")
        success = run_command(
            "Ejecutando feature engineering",
            f'python -c "import sys; sys.stdout.reconfigure(encoding=\'utf-8\'); exec(open(\'ft_engineering.py\', encoding=\'utf-8\').read())"',
            cwd=src_dir
        )
        if not success:
            print("[ERROR] Error en feature engineering. Abortando...")
            return 1
    else:
        print_header("PASO 1: Feature Engineering")
        print("[OK] Datos ya procesados - Saltando este paso...")
        print(f"     Encontrados: {len(data_files)} archivos necesarios")
    
    # 2. Entrenamiento de múltiples modelos
    print_header("PASO 2: Entrenamiento de Multiples Modelos")
    print("[INFO] Se entrenaran 5 modelos diferentes:")
    print("        - Logistic Regression")
    print("        - Random Forest") 
    print("        - XGBoost")
    print("        - LightGBM")
    print("        - Gradient Boosting")
    print("")
    success = run_command(
        "Entrenando y comparando modelos",
        "python train_multiple_models.py",
        cwd=src_dir
    )
    if not success:
        print("[ERROR] Error en entrenamiento. Abortando...")
        return 1
    
    # 3. Monitoreo
    print_header("PASO 3: Monitoreo de Datos")
    print("[*] Ejecutando monitoreo de drift (puede tomar unos minutos)...")
    
    # Crear wrapper para ejecutar con encoding UTF-8
    wrapper_code = """
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
exec(open('model_monitoring.py', encoding='utf-8').read())
"""
    
    wrapper_path = src_dir / "run_monitoring_wrapper.py"
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    success = run_command(
        "Analizando drift en datos",
        "python run_monitoring_wrapper.py",
        cwd=src_dir
    )
    
    # Limpiar archivo temporal
    if wrapper_path.exists():
        wrapper_path.unlink()
    
    if not success:
        print("[ERROR] Error en monitoreo. Abortando...")
        return 1
    
    # Resumen final
    print_header("PIPELINE COMPLETADO EXITOSAMENTE")
    print("Resultados generados:")
    print("   - Modelo: models/xgboost_model.pkl")
    print("   - Metricas: models/model_metrics.pkl")
    print("   - Monitoreo: outputs/monitoring/")
    
    # Abrir dashboard automáticamente
    print_header("Iniciando Dashboard Interactivo")
    print("Abriendo Streamlit dashboard en http://localhost:8501")
    print("Presiona Ctrl+C para detener el servidor\n")
    subprocess.run("streamlit run app_monitoring.py", shell=True, cwd=base_dir)
    
    print("\n" + "="*80 + "\n")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
