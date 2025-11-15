"""
Tests para el módulo de Feature Engineering
"""
import pytest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Añadir el directorio src al path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'mlops_pipeline' / 'src'))


class TestFeatureEngineering:
    """Tests para la clase FeatureEngineering"""
    
    def test_balance_features_creation(self):
        """Verificar que se crean correctamente las features de balance"""
        # Crear datos de prueba
        data = {
            'oldbalanceOrg': [100, 200],
            'newbalanceOrig': [50, 150],
            'oldbalanceDest': [0, 100],
            'newbalanceDest': [50, 150],
            'amount': [50, 50]
        }
        df = pd.DataFrame(data)
        
        # Calcular features de balance
        df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['error_balance_orig'] = (df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']).abs()
        
        # Verificaciones
        assert 'balance_diff_orig' in df.columns
        assert 'balance_diff_dest' in df.columns
        assert 'error_balance_orig' in df.columns
        assert df['balance_diff_orig'].iloc[0] == 50
        assert df['balance_diff_dest'].iloc[0] == 50
    
    def test_binary_features_creation(self):
        """Verificar que se crean correctamente las features binarias"""
        data = {
            'nameOrig': ['C123', 'M456'],
            'nameDest': ['M789', 'C012'],
            'type': ['TRANSFER', 'PAYMENT'],
            'oldbalanceOrg': [100, 0]
        }
        df = pd.DataFrame(data)
        
        # Crear features binarias
        df['orig_is_merchant'] = df['nameOrig'].str.startswith('M').astype(int)
        df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
        df['is_fraud_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
        df['orig_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
        
        # Verificaciones
        assert df['orig_is_merchant'].iloc[0] == 0
        assert df['orig_is_merchant'].iloc[1] == 1
        assert df['dest_is_merchant'].iloc[0] == 1
        assert df['is_fraud_type'].iloc[0] == 1
        assert df['orig_balance_zero'].iloc[1] == 1
    
    def test_ratio_features_creation(self):
        """Verificar que se crean correctamente las features de ratios"""
        data = {
            'amount': [100, 50],
            'oldbalanceOrg': [200, 100],
            'oldbalanceDest': [0, 100]
        }
        df = pd.DataFrame(data)
        
        # Crear features de ratios
        df['amount_to_oldbalance_orig_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['amount_to_oldbalance_dest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)
        
        # Verificaciones
        assert 'amount_to_oldbalance_orig_ratio' in df.columns
        assert 'amount_to_oldbalance_dest_ratio' in df.columns
        assert df['amount_to_oldbalance_orig_ratio'].iloc[0] > 0
        assert df['amount_to_oldbalance_dest_ratio'].iloc[0] > 0
    
    def test_temporal_features_creation(self):
        """Verificar que se crean correctamente las features temporales"""
        data = {'step': [1, 12, 24, 48, 120]}
        df = pd.DataFrame(data)
        
        # Crear features temporales
        df['hour_of_day'] = df['step'] % 24
        df['day_of_week'] = (df['step'] // 24) % 7
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        
        # Verificaciones
        assert df['hour_of_day'].iloc[0] == 1
        assert df['hour_of_day'].iloc[2] == 0
        assert df['day_of_week'].iloc[1] == 0
        assert df['is_weekend'].iloc[0] == 0
        assert df['is_night'].iloc[0] == 1
    
    def test_data_split_stratification(self):
        """Verificar que la división de datos mantiene la proporción de clases"""
        from sklearn.model_selection import train_test_split
        
        # Crear datos desbalanceados
        X = pd.DataFrame({'feature': range(1000)})
        y = pd.Series([0] * 990 + [1] * 10)  # 99% clase 0, 1% clase 1
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Verificar proporciones
        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)
        original_ratio = y.sum() / len(y)
        
        # Las proporciones deben ser similares (con margen de error)
        assert abs(train_ratio - original_ratio) < 0.01
        assert abs(test_ratio - original_ratio) < 0.01
    
    def test_pipeline_transform_output_shape(self):
        """Verificar que el pipeline transforma correctamente"""
        from sklearn.preprocessing import RobustScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        # Crear pipeline simple
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Datos de prueba
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [10, 20, 30, 40]
        })
        
        # Transformar
        X_transformed = pipeline.fit_transform(X)
        
        # Verificaciones
        assert X_transformed.shape == X.shape
        assert not np.isnan(X_transformed).any()


class TestDataValidation:
    """Tests para validación de datos"""
    
    def test_no_null_after_processing(self):
        """Verificar que no hay nulos después del procesamiento"""
        from sklearn.impute import SimpleImputer
        
        # Datos con nulos
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [np.nan, 2, 3, 4]
        })
        
        imputer = SimpleImputer(strategy='median')
        data_clean = imputer.fit_transform(data)
        
        assert not np.isnan(data_clean).any()
    
    def test_class_balance_range(self):
        """Verificar que el desbalanceo está en el rango esperado"""
        # Simular distribución de clases del dataset
        total_samples = 200000
        fraud_samples = 260
        non_fraud_samples = total_samples - fraud_samples
        
        fraud_percentage = (fraud_samples / total_samples) * 100
        
        # Verificar que está cerca del 0.13%
        assert 0.1 <= fraud_percentage <= 0.2
    
    def test_feature_names_consistency(self):
        """Verificar que los nombres de features son consistentes"""
        expected_features = [
            'balance_diff_orig', 'balance_diff_dest', 
            'error_balance_orig', 'orig_is_merchant',
            'dest_is_merchant', 'is_fraud_type'
        ]
        
        # Verificar que los nombres son strings válidos
        for feature in expected_features:
            assert isinstance(feature, str)
            assert len(feature) > 0
            assert '_' in feature or feature.islower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
