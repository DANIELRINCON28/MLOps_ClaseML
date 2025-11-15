"""
Script de Testing para la API de Detección de Fraude
=====================================================

Este script prueba todos los endpoints de la API.

Uso:
    python test_api.py
"""

import requests
import json
import time
from pathlib import Path

# Configuración
API_URL = "http://localhost:8000"
TIMEOUT = 10

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")


# ============================================================================
# TESTS
# ============================================================================

def test_root():
    """Test endpoint raíz"""
    print("\n" + "="*80)
    print("TEST 1: Endpoint Raíz (GET /)")
    print("="*80)
    
    try:
        response = requests.get(f"{API_URL}/", timeout=TIMEOUT)
        
        if response.status_code == 200:
            print_success(f"Status: {response.status_code}")
            data = response.json()
            print_info(f"API: {data.get('api')}")
            print_info(f"Versión: {data.get('version')}")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_health():
    """Test health check"""
    print("\n" + "="*80)
    print("TEST 2: Health Check (GET /health)")
    print("="*80)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        
        if response.status_code == 200:
            print_success(f"Status: {response.status_code}")
            data = response.json()
            
            if data.get('status') == 'healthy':
                print_success(f"Estado: {data.get('status')}")
            else:
                print_warning(f"Estado: {data.get('status')}")
            
            print_info(f"Modelo cargado: {data.get('model_loaded')}")
            print(json.dumps(data, indent=2))
            return data.get('model_loaded', False)
        else:
            print_error(f"Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_model_info():
    """Test información del modelo"""
    print("\n" + "="*80)
    print("TEST 3: Información del Modelo (GET /model/info)")
    print("="*80)
    
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=TIMEOUT)
        
        if response.status_code == 200:
            print_success(f"Status: {response.status_code}")
            data = response.json()
            print_info(f"Modelo: {data.get('model_name')}")
            print_info(f"Entrenado: {data.get('trained_on')}")
            print_info(f"Features: {data.get('features_count')}")
            
            metrics = data.get('metrics', {})
            if metrics:
                print("\nMétricas del modelo:")
                for metric, value in metrics.items():
                    print(f"  • {metric}: {value:.4f}")
            
            print("\n" + json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Status: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_predict_single():
    """Test predicción individual"""
    print("\n" + "="*80)
    print("TEST 4: Predicción Individual (POST /predict)")
    print("="*80)
    
    # Ejemplo de transacción legítima
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
    
    print("Transacción de prueba:")
    print(json.dumps(transaction, indent=2))
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            timeout=TIMEOUT
        )
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            print_success(f"Status: {response.status_code}")
            data = response.json()
            
            if data.get('is_fraud') == 1:
                print_error(f"Predicción: FRAUDE ⚠️")
            else:
                print_success(f"Predicción: LEGÍTIMO ✅")
            
            print_info(f"Probabilidad: {data.get('fraud_probability'):.4f}")
            print_info(f"Nivel de riesgo: {data.get('risk_level')}")
            print_info(f"Tiempo de respuesta: {elapsed:.2f}ms")
            
            print("\nRespuesta completa:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Status: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_predict_batch():
    """Test predicción por lotes"""
    print("\n" + "="*80)
    print("TEST 5: Predicción Batch (POST /predict/batch)")
    print("="*80)
    
    # Lote con 3 transacciones (1 legítima, 2 sospechosas)
    batch = {
        "transactions": [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 9839.64,
                "nameOrig": "C1231006815",
                "oldbalanceOrg": 170136.0,
                "newbalanceOrig": 160296.36,
                "nameDest": "M1979787155",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0
            },
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C840083671",
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C38997010",
                "oldbalanceDest": 21182.0,
                "newbalanceDest": 0.0
            },
            {
                "step": 5,
                "type": "CASH_OUT",
                "amount": 50000.0,
                "nameOrig": "C123456789",
                "oldbalanceOrg": 50000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C987654321",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 50000.0
            }
        ]
    }
    
    print(f"Probando batch de {len(batch['transactions'])} transacciones...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=batch,
            timeout=TIMEOUT
        )
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            print_success(f"Status: {response.status_code}")
            data = response.json()
            
            print_info(f"Total procesadas: {data.get('total_transactions')}")
            print_info(f"Fraudes detectados: {data.get('frauds_detected')}")
            print_info(f"Tasa de fraude: {data.get('fraud_rate'):.2f}%")
            print_info(f"Tiempo de procesamiento API: {data.get('processing_time_ms'):.2f}ms")
            print_info(f"Tiempo total request: {elapsed:.2f}ms")
            
            print("\nPredicciones individuales:")
            for i, pred in enumerate(data.get('predictions', []), 1):
                status = "FRAUDE ⚠️" if pred['is_fraud'] == 1 else "LEGÍTIMO ✅"
                print(f"  {i}. {status} - Prob: {pred['fraud_probability']:.4f} - Riesgo: {pred['risk_level']}")
            
            print("\nRespuesta completa:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print_error(f"Status: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_predict_csv():
    """Test predicción desde CSV"""
    print("\n" + "="*80)
    print("TEST 6: Predicción desde CSV (POST /predict/csv)")
    print("="*80)
    
    # Crear CSV temporal
    csv_content = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest
1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0
1,TRANSFER,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0
5,CASH_OUT,50000.0,C123456789,50000.0,0.0,C987654321,0.0,50000.0"""
    
    csv_file = Path("test_transactions.csv")
    csv_file.write_text(csv_content)
    
    print(f"CSV de prueba creado: {csv_file}")
    print("Contenido:")
    print(csv_content)
    
    try:
        with open(csv_file, 'rb') as f:
            files = {'file': f}
            
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/predict/csv",
                files=files,
                timeout=TIMEOUT
            )
            elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            print_success(f"Status: {response.status_code}")
            data = response.json()
            
            print_info(f"Total procesadas: {data.get('total_transactions')}")
            print_info(f"Fraudes detectados: {data.get('frauds_detected')}")
            print_info(f"Tasa de fraude: {data.get('fraud_rate'):.2f}%")
            print_info(f"Tiempo de procesamiento: {data.get('processing_time_ms'):.2f}ms")
            print_info(f"Tiempo total request: {elapsed:.2f}ms")
            
            print("\nPredicciones:")
            for i, pred in enumerate(data.get('predictions', []), 1):
                status = "FRAUDE ⚠️" if pred['is_fraud'] == 1 else "LEGÍTIMO ✅"
                print(f"  {i}. {status} - Prob: {pred['fraud_probability']:.4f}")
            
            # Limpiar archivo
            csv_file.unlink()
            return True
        else:
            print_error(f"Status: {response.status_code}")
            print(response.text)
            csv_file.unlink()
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        if csv_file.exists():
            csv_file.unlink()
        return False


def test_invalid_transaction():
    """Test validación de datos inválidos"""
    print("\n" + "="*80)
    print("TEST 7: Validación de Datos Inválidos")
    print("="*80)
    
    # Transacción con tipo inválido
    invalid_transaction = {
        "step": 1,
        "type": "INVALID_TYPE",  # Tipo inválido
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    
    print("Transacción inválida (tipo incorrecto):")
    print(json.dumps(invalid_transaction, indent=2))
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=invalid_transaction,
            timeout=TIMEOUT
        )
        
        if response.status_code == 422:  # Unprocessable Entity
            print_success("✅ Validación funcionando correctamente (422 esperado)")
            print("Error devuelto:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print_warning(f"Status inesperado: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def run_all_tests():
    """Ejecuta todos los tests"""
    print("\n" + "="*80)
    print("🧪 TESTS DE LA API DE DETECCIÓN DE FRAUDE")
    print("="*80)
    print(f"URL: {API_URL}")
    print(f"Timeout: {TIMEOUT}s")
    
    results = {}
    
    # Verificar que la API esté corriendo
    try:
        requests.get(f"{API_URL}/", timeout=2)
    except Exception:
        print_error(f"\n❌ ERROR: La API no está corriendo en {API_URL}")
        print_info("Inicia la API con: python api/main.py")
        return
    
    # Ejecutar tests
    results['root'] = test_root()
    results['health'] = test_health()
    results['model_info'] = test_model_info()
    results['predict_single'] = test_predict_single()
    results['predict_batch'] = test_predict_batch()
    results['predict_csv'] = test_predict_csv()
    results['invalid_data'] = test_invalid_transaction()
    
    # Resumen
    print("\n" + "="*80)
    print("📊 RESUMEN DE TESTS")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "-"*80)
    print(f"Total: {passed}/{total} tests pasados ({passed/total*100:.1f}%)")
    print("="*80)
    
    if passed == total:
        print_success("\n🎉 ¡Todos los tests pasaron exitosamente!")
    else:
        print_warning(f"\n⚠️ {total - passed} test(s) fallaron")


if __name__ == "__main__":
    run_all_tests()
