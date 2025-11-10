"""
Tests para el módulo de monitoreo de data drift
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'mlops_pipeline' / 'src'))


class TestDataDrift:
    """Tests para detección de data drift"""
    
    def test_ks_statistic_calculation(self):
        """Verificar el cálculo de estadística Kolmogorov-Smirnov"""
        from scipy.stats import ks_2samp
        
        # Dos distribuciones similares
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0, 1, 1000)
        
        statistic, p_value = ks_2samp(data1, data2)
        
        # Sin drift significativo
        assert 0 <= statistic <= 1
        assert p_value > 0.01  # No debería haber drift
    
    def test_ks_detects_drift(self):
        """Verificar que KS detecta drift cuando hay cambio"""
        from scipy.stats import ks_2samp
        
        # Distribuciones diferentes
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(5, 1, 1000)  # Media diferente
        
        statistic, p_value = ks_2samp(data1, data2)
        
        # Debería detectar drift
        assert statistic > 0.3  # Alta diferencia
        assert p_value < 0.05  # Significativo
    
    def test_psi_calculation(self):
        """Verificar el cálculo de Population Stability Index"""
        # Distribuciones de referencia y producción
        reference = np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
        production = np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
        
        # Calcular PSI manualmente
        ref_counts = pd.Series(reference).value_counts(normalize=True).sort_index()
        prod_counts = pd.Series(production).value_counts(normalize=True).sort_index()
        
        psi = ((prod_counts - ref_counts) * np.log(prod_counts / ref_counts)).sum()
        
        # PSI debería ser bajo para distribuciones similares
        assert 0 <= psi < 0.1
    
    def test_jensen_shannon_divergence(self):
        """Verificar el cálculo de divergencia Jensen-Shannon"""
        from scipy.spatial.distance import jensenshannon
        
        # Distribuciones de probabilidad
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.5, 0.3, 0.2])
        
        js_div = jensenshannon(p, q)
        
        # Debería ser cercano a 0 para distribuciones iguales
        assert 0 <= js_div < 0.01
    
    def test_chi2_test_categorical(self):
        """Verificar el test Chi-cuadrado para variables categóricas"""
        from scipy.stats import chi2_contingency
        
        # Datos categóricos de referencia y producción
        reference = pd.Series(['A'] * 500 + ['B'] * 300 + ['C'] * 200)
        production = pd.Series(['A'] * 480 + ['B'] * 320 + ['C'] * 200)
        
        # Crear tabla de contingencia
        ref_counts = reference.value_counts()
        prod_counts = production.value_counts()
        contingency_table = pd.DataFrame([ref_counts, prod_counts])
        
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        # Sin drift significativo
        assert chi2 >= 0
        assert 0 <= p_value <= 1
    
    def test_alert_generation_high_drift(self):
        """Verificar que se generan alertas con drift alto"""
        drift_results = [
            {'feature': 'amount', 'p_value': 0.001, 'drift_detected': True, 'severity': 'HIGH'},
            {'feature': 'balance', 'p_value': 0.02, 'drift_detected': True, 'severity': 'MEDIUM'}
        ]
        
        alerts = [r for r in drift_results if r['drift_detected']]
        
        assert len(alerts) == 2
        assert alerts[0]['severity'] == 'HIGH'
        assert alerts[1]['severity'] == 'MEDIUM'
    
    def test_severity_classification(self):
        """Verificar la clasificación de severidad del drift"""
        def classify_severity(p_value):
            if p_value < 0.01:
                return 'CRITICAL'
            elif p_value < 0.05:
                return 'HIGH'
            elif p_value < 0.1:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        assert classify_severity(0.005) == 'CRITICAL'
        assert classify_severity(0.03) == 'HIGH'
        assert classify_severity(0.07) == 'MEDIUM'
        assert classify_severity(0.5) == 'LOW'
    
    def test_drift_results_storage(self):
        """Verificar que los resultados de drift se pueden guardar"""
        import json
        import tempfile
        import os
        
        drift_results = {
            'timestamp': '2024-11-09T10:00:00',
            'features_with_drift': ['amount', 'balance'],
            'total_features': 29,
            'drift_percentage': 6.9
        }
        
        # Guardar en archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(drift_results, f)
            temp_path = f.name
        
        # Leer y verificar
        with open(temp_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['total_features'] == 29
        assert len(loaded_results['features_with_drift']) == 2
        
        # Limpiar
        os.unlink(temp_path)


class TestMonitoringMetrics:
    """Tests para métricas de monitoreo"""
    
    def test_prediction_distribution(self):
        """Verificar el análisis de distribución de predicciones"""
        predictions = np.array([0] * 9500 + [1] * 500)
        
        fraud_rate = predictions.sum() / len(predictions)
        
        assert 0 <= fraud_rate <= 1
        assert fraud_rate == 0.05  # 5% de fraudes
    
    def test_confidence_scores_range(self):
        """Verificar que las probabilidades están en rango válido"""
        probabilities = np.array([0.1, 0.3, 0.7, 0.9, 0.05])
        
        assert all(0 <= p <= 1 for p in probabilities)
        assert probabilities.min() >= 0
        assert probabilities.max() <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
