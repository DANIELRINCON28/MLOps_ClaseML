"""
Tests para el módulo de entrenamiento de modelos
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'mlops_pipeline' / 'src'))


class TestModelTraining:
    """Tests para el entrenamiento de modelos"""
    
    def test_smote_increases_minority_class(self):
        """Verificar que SMOTE aumenta la clase minoritaria"""
        from imblearn.over_sampling import SMOTE
        
        # Datos desbalanceados
        X = np.random.rand(1000, 5)
        y = np.array([0] * 950 + [1] * 50)
        
        # Aplicar SMOTE
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Verificar que aumentó la clase minoritaria
        assert y_resampled.sum() > y.sum()
        assert len(y_resampled) > len(y)
    
    def test_model_can_be_trained(self):
        """Verificar que se puede entrenar un modelo básico"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Datos de prueba
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Verificar que puede predecir
        X_test = np.random.rand(20, 10)
        predictions = model.predict(X_test)
        
        assert len(predictions) == 20
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_evaluation_metrics(self):
        """Verificar que se pueden calcular métricas de evaluación"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Predicciones simuladas
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1])
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Verificaciones
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
    
    def test_roc_auc_calculation(self):
        """Verificar que se puede calcular ROC-AUC"""
        from sklearn.metrics import roc_auc_score
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.7, 0.6, 0.3, 0.8, 0.2, 0.9])
        
        roc_auc = roc_auc_score(y_true, y_scores)
        
        assert 0 <= roc_auc <= 1
        assert roc_auc > 0.5  # Mejor que random
    
    def test_confusion_matrix_shape(self):
        """Verificar la forma de la matriz de confusión"""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
    
    def test_model_serialization(self):
        """Verificar que se puede guardar y cargar un modelo"""
        import pickle
        from sklearn.linear_model import LogisticRegression
        import tempfile
        import os
        
        # Entrenar modelo simple
        model = LogisticRegression(random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        # Guardar modelo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(model, f)
            temp_path = f.name
        
        # Cargar modelo
        with open(temp_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Verificar que funciona
        predictions = loaded_model.predict(X[:10])
        assert len(predictions) == 10
        
        # Limpiar
        os.unlink(temp_path)


class TestModelComparison:
    """Tests para la comparación de modelos"""
    
    def test_results_dataframe_creation(self):
        """Verificar que se puede crear un DataFrame de resultados"""
        results = {
            'Model': ['Model_A', 'Model_B'],
            'accuracy': [0.95, 0.93],
            'precision': [0.90, 0.88],
            'recall': [0.85, 0.92],
            'f1_score': [0.87, 0.90],
            'roc_auc': [0.96, 0.94]
        }
        
        df = pd.DataFrame(results)
        df_sorted = df.sort_values('roc_auc', ascending=False)
        
        assert len(df_sorted) == 2
        assert df_sorted.iloc[0]['Model'] == 'Model_A'
        assert 'roc_auc' in df_sorted.columns
    
    def test_best_model_selection(self):
        """Verificar que se selecciona el mejor modelo correctamente"""
        results = {
            'Logistic_Regression': {'roc_auc': 0.85},
            'Random_Forest': {'roc_auc': 0.95},
            'XGBoost': {'roc_auc': 0.92}
        }
        
        best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
        
        assert best_model_name == 'Random_Forest'
        assert results[best_model_name]['roc_auc'] == 0.95


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
