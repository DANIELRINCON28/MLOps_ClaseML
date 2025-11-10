"""
Script simplificado para ejecutar todo el pipeline MLOps directamente
======================================================================
Sin usar subprocess para evitar problemas en Windows
"""

import sys
import os
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def main():
    print_header("PIPELINE MLOPS COMPLETO - DETECCION DE FRAUDE")
    
    # Directorio base
    base_dir = Path(__file__).parent
    src_dir = base_dir / "mlops_pipeline" / "src"
    
    # Cambiar al directorio src para ejecución
    os.chdir(src_dir)
    
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
        try:
            exec(open('ft_engineering.py', encoding='utf-8').read())
            print("[OK] Feature engineering completado")
        except Exception as e:
            print(f"[ERROR] Error en feature engineering: {e}")
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
    
    try:
        exec(open('train_multiple_models.py', encoding='utf-8').read())
        print("[OK] Entrenamiento de modelos completado")
    except Exception as e:
        print(f"[ERROR] Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 3. Monitoreo
    print_header("PASO 3: Monitoreo de Datos")
    print("[*] Ejecutando monitoreo de drift...")
    
    try:
        # Importar y ejecutar el monitoreo
        import subprocess
        result = subprocess.run(
            [sys.executable, 'model_monitoring.py'],
            cwd=src_dir,
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print("[OK] Monitoreo completado")
        else:
            print(f"[ERROR] Error en monitoreo (código: {result.returncode})")
            return 1
    except Exception as e:
        print(f"[ERROR] Error en monitoreo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Resumen final
    print_header("PIPELINE COMPLETADO EXITOSAMENTE")
    print("Resultados generados:")
    print("   - Modelos: models/")
    print("   - Metricas: outputs/")
    print("   - Monitoreo: outputs/monitoring/")
    
    # Regresar al directorio base
    os.chdir(base_dir)
    
    # Abrir dashboard automáticamente
    print_header("Iniciando Dashboard Interactivo")
    print("Abriendo Streamlit dashboard en http://localhost:8501")
    print("Presiona Ctrl+C para detener el servidor\n")
    
    import subprocess
    streamlit_cmd = str(base_dir / "MLOPS_FINAL-venv" / "Scripts" / "streamlit.exe")
    subprocess.run([streamlit_cmd, "run", "app_monitoring.py"], shell=False, cwd=base_dir)
    
    print("\n" + "="*80 + "\n")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[!] Ejecucion interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
