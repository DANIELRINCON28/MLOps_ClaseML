"""
Script de Verificación del Entorno
===================================

Este script verifica que todas las dependencias estén instaladas correctamente
y que el entorno esté listo para ejecutar el pipeline de MLOps.

Ejecutar con:
    python check_environment.py
"""

import sys

print("=" * 80)
print("VERIFICACIÓN DEL ENTORNO - Pipeline MLOps de Detección de Fraude")
print("=" * 80)

# Lista de dependencias requeridas
dependencies = {
    'pandas': 'Manipulación de datos',
    'numpy': 'Operaciones numéricas',
    'sklearn': 'Machine Learning',
    'xgboost': 'Gradient Boosting',
    'lightgbm': 'Gradient Boosting ligero',
    'imblearn': 'Balanceo de clases',
    'matplotlib': 'Visualización',
    'seaborn': 'Visualización estadística',
    'scipy': 'Operaciones científicas',
    'pickle': 'Serialización (built-in)',
    'json': 'Manejo de JSON (built-in)'
}

print("\n🔍 Verificando instalación de librerías...\n")

missing_packages = []
installed_packages = []

for package, description in dependencies.items():
    try:
        if package == 'sklearn':
            import sklearn
            version = sklearn.__version__
        elif package == 'imblearn':
            import imblearn
            version = imblearn.__version__
        else:
            module = __import__(package)
            version = getattr(module, '__version__', 'N/A')
        
        installed_packages.append(package)
        print(f"✅ {package:<15} v{version:<10} - {description}")
    except ImportError:
        missing_packages.append(package)
        print(f"❌ {package:<15} {'NO INSTALADO':<10} - {description}")

# Resumen
print("\n" + "=" * 80)
print("RESUMEN DE VERIFICACIÓN")
print("=" * 80)

print(f"\n✅ Paquetes instalados: {len(installed_packages)}/{len(dependencies)}")
print(f"❌ Paquetes faltantes: {len(missing_packages)}")

if missing_packages:
    print("\n⚠️ ADVERTENCIA: Algunos paquetes no están instalados")
    print("\nPara instalar los paquetes faltantes, ejecuta:")
    print(f"\npip install {' '.join(missing_packages)}")
    print("\nO instala todas las dependencias con:")
    print("pip install -r requirements.txt")
else:
    print("\n🎉 ¡Todas las dependencias están instaladas correctamente!")

# Verificar versión de Python
print("\n" + "=" * 80)
print("VERIFICACIÓN DE PYTHON")
print("=" * 80)

python_version = sys.version_info
print(f"\n🐍 Versión de Python: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major >= 3 and python_version.minor >= 8:
    print("✅ Versión de Python compatible (>= 3.8)")
else:
    print("⚠️ ADVERTENCIA: Se recomienda Python 3.8 o superior")

# Verificar estructura de directorios
print("\n" + "=" * 80)
print("VERIFICACIÓN DE ESTRUCTURA DE DIRECTORIOS")
print("=" * 80)

import os

directories = {
    'data/processed': 'Datos procesados',
    'models': 'Modelos entrenados',
    'outputs': 'Gráficos y reportes',
    'mlops_pipeline/src': 'Scripts del pipeline'
}

print("\n📁 Verificando directorios...\n")

for directory, description in directories.items():
    if os.path.exists(directory):
        print(f"✅ {directory:<25} - {description}")
    else:
        print(f"⚠️ {directory:<25} - {description} (será creado automáticamente)")

# Verificar archivo de datos
print("\n" + "=" * 80)
print("VERIFICACIÓN DE DATOS")
print("=" * 80)

data_file = 'Base_datos.csv'
print(f"\n📊 Verificando archivo de datos: {data_file}")

if os.path.exists(data_file):
    file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
    print(f"✅ Archivo encontrado - Tamaño: {file_size_mb:.2f} MB")
else:
    print(f"❌ Archivo no encontrado: {data_file}")
    print("   Por favor, asegúrate de que Base_datos.csv esté en el directorio raíz")

# Resumen final
print("\n" + "=" * 80)
print("ESTADO GENERAL DEL ENTORNO")
print("=" * 80)

all_checks_passed = (
    len(missing_packages) == 0 and
    python_version.major >= 3 and python_version.minor >= 8 and
    os.path.exists(data_file)
)

if all_checks_passed:
    print("\n🎉 ¡EL ENTORNO ESTÁ LISTO PARA USAR!")
    print("\n📖 Próximos pasos:")
    print("   1. Ejecutar: jupyter notebook mlops_pipeline/src/Cargar_datos.ipynb")
    print("   2. Ejecutar: jupyter notebook mlops_pipeline/src/Comprension_eda_completo.ipynb")
    print("   3. Ejecutar: python mlops_pipeline/src/ft_engineering.py")
    print("   4. Ejecutar: python mlops_pipeline/src/model_training_evaluation.py")
    print("\n   O consulta INSTRUCCIONES_EJECUCION.md para más detalles")
else:
    print("\n⚠️ ATENCIÓN: Algunos problemas detectados")
    print("\nPor favor, revisa los mensajes anteriores y corrige los problemas")
    if missing_packages:
        print("\n1. Instala las dependencias faltantes:")
        print("   pip install -r requirements.txt")
    if not os.path.exists(data_file):
        print("\n2. Asegúrate de tener el archivo Base_datos.csv en el directorio raíz")

print("\n" + "=" * 80)
print("Verificación completada")
print("=" * 80)
