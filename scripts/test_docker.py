#!/usr/bin/env python3
"""
Script de Verificación de Docker
=================================

Verifica que la imagen Docker esté correctamente construida y funcionando.

Uso:
    python scripts/test_docker.py
"""

import requests
import time
import sys
import subprocess
import json

# Colores
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(msg):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.YELLOW}→ {msg}{Colors.END}")

def check_docker():
    """Verifica si Docker está instalado"""
    print_header("Verificando Docker")
    
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print_success(f"Docker instalado: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Docker no está instalado o no está en PATH")
        print_info("Instala Docker Desktop desde: https://www.docker.com/products/docker-desktop")
        return False

def check_docker_compose():
    """Verifica si Docker Compose está disponible"""
    print_header("Verificando Docker Compose")
    
    try:
        result = subprocess.run(
            ['docker-compose', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print_success(f"Docker Compose instalado: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Docker Compose no está instalado")
        return False

def check_image_exists():
    """Verifica si la imagen existe"""
    print_header("Verificando Imagen Docker")
    
    try:
        result = subprocess.run(
            ['docker', 'images', 'fraud-detection-api', '--format', '{{.Repository}}:{{.Tag}}'],
            capture_output=True,
            text=True,
            check=True
        )
        
        if 'fraud-detection-api' in result.stdout:
            print_success(f"Imagen encontrada: {result.stdout.strip()}")
            return True
        else:
            print_error("Imagen 'fraud-detection-api' no encontrada")
            print_info("Construye la imagen con: docker build -t fraud-detection-api:latest .")
            return False
    except subprocess.CalledProcessError as e:
        print_error(f"Error al verificar imagen: {e}")
        return False

def check_container_running():
    """Verifica si el contenedor está corriendo"""
    print_header("Verificando Contenedor")
    
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=fraud-api', '--format', '{{.Names}}\t{{.Status}}'],
            capture_output=True,
            text=True,
            check=True
        )
        
        if 'fraud-api' in result.stdout:
            status = result.stdout.split('\t')[1] if '\t' in result.stdout else 'Unknown'
            print_success(f"Contenedor 'fraud-api' corriendo: {status}")
            return True
        else:
            print_error("Contenedor 'fraud-api' no está corriendo")
            print_info("Inicia el contenedor con: docker-compose up -d")
            return False
    except subprocess.CalledProcessError as e:
        print_error(f"Error al verificar contenedor: {e}")
        return False

def test_api_health():
    """Prueba el endpoint de health"""
    print_header("Probando API - Health Check")
    
    url = "http://localhost:8000/health"
    
    print_info("Esperando que la API esté lista...")
    max_retries = 30
    
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=2)
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"API respondiendo en {url}")
                print_success(f"Estado: {data.get('status')}")
                print_success(f"Modelo cargado: {data.get('model_loaded')}")
                print_success(f"Nombre del modelo: {data.get('model_name')}")
                
                print("\nRespuesta completa:")
                print(json.dumps(data, indent=2))
                return True
            else:
                print_error(f"Status code inesperado: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print_info(f"Intento {i+1}/{max_retries} - API no disponible, reintentando...")
                time.sleep(2)
            else:
                print_error(f"No se pudo conectar a {url} después de {max_retries} intentos")
                print_info("Verifica que el contenedor esté corriendo: docker ps")
                print_info("Ver logs: docker logs fraud-api")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    return False

def test_api_prediction():
    """Prueba una predicción"""
    print_header("Probando API - Predicción")
    
    url = "http://localhost:8000/predict"
    
    transaction = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    
    print_info("Enviando transacción de prueba...")
    print(json.dumps(transaction, indent=2))
    
    try:
        response = requests.post(url, json=transaction, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Predicción exitosa")
            print_success(f"Es fraude: {'Sí' if data['is_fraud'] == 1 else 'No'}")
            print_success(f"Probabilidad: {data['fraud_probability']:.4f}")
            print_success(f"Nivel de riesgo: {data['risk_level']}")
            
            print("\nRespuesta completa:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Error en predicción: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_api_docs():
    """Verifica que la documentación esté disponible"""
    print_header("Verificando Documentación")
    
    url = "http://localhost:8000/docs"
    
    try:
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print_success(f"Documentación disponible en {url}")
            return True
        else:
            print_error(f"Error al acceder a docs: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def check_logs():
    """Muestra los últimos logs del contenedor"""
    print_header("Últimos Logs del Contenedor")
    
    try:
        result = subprocess.run(
            ['docker', 'logs', '--tail', '20', 'fraud-api'],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("\nErrores:")
            print(result.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Error al obtener logs: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("    VERIFICACIÓN DE DOCKER - FRAUD DETECTION API")
    print("="*70 + "\n")
    
    results = {
        "Docker instalado": check_docker(),
        "Docker Compose instalado": check_docker_compose(),
        "Imagen construida": check_image_exists(),
        "Contenedor corriendo": check_container_running(),
    }
    
    # Solo probar API si el contenedor está corriendo
    if results["Contenedor corriendo"]:
        results["API Health OK"] = test_api_health()
        results["API Prediction OK"] = test_api_prediction()
        results["API Docs OK"] = test_api_docs()
        check_logs()
    
    # Resumen
    print_header("RESUMEN DE VERIFICACIÓN")
    
    passed = 0
    total = len(results)
    
    for test, result in results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"{status} - {test}")
        if result:
            passed += 1
    
    print(f"\n{Colors.BLUE}Total: {passed}/{total} verificaciones exitosas ({passed/total*100:.1f}%){Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}¡Todo funcionando correctamente!{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
        
        print("Accede a:")
        print(f"  • API: http://localhost:8000")
        print(f"  • Docs: http://localhost:8000/docs")
        print(f"  • Health: http://localhost:8000/health")
        
        return 0
    else:
        print(f"\n{Colors.RED}{'='*70}{Colors.END}")
        print(f"{Colors.RED}Algunas verificaciones fallaron{Colors.END}")
        print(f"{Colors.RED}{'='*70}{Colors.END}\n")
        
        print("Pasos sugeridos:")
        if not results["Docker instalado"]:
            print("  1. Instala Docker Desktop")
        if not results["Imagen construida"]:
            print("  2. Construye la imagen: docker build -t fraud-detection-api:latest .")
        if not results["Contenedor corriendo"]:
            print("  3. Inicia el contenedor: docker-compose up -d")
        
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrumpido por usuario{Colors.END}")
        sys.exit(130)
