#!/usr/bin/env python
"""
Script principal de ejecución del proyecto MLOps
Ejecuta el pipeline completo de MLOps desde la raíz del proyecto
"""

import sys
import os
from pathlib import Path

# Agregar el directorio actual al path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Importar y ejecutar el pipeline
from mlops_pipeline.src.run_full_pipeline import main

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 INICIANDO PIPELINE MLOPS")
    print("=" * 60)
    print(f"📁 Directorio del proyecto: {PROJECT_ROOT}")
    print("=" * 60)
    
    try:
        main()
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR EN EL PIPELINE: {e}")
        print("=" * 60)
        sys.exit(1)
