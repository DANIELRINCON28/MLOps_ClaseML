"""
Tests de utilidades y helpers del proyecto
"""
import pytest
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path


class TestDataLoading:
    """Tests para carga de datos"""
    
    def test_csv_loading(self):
        """Verificar que se puede cargar un CSV"""
        import tempfile
        
        # Crear CSV temporal
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        # Cargar CSV
        loaded_data = pd.read_csv(temp_path)
        
        assert len(loaded_data) == 3
        assert 'col1' in loaded_data.columns
        
        # Limpiar
        os.unlink(temp_path)
    
    def test_pickle_save_load(self):
        """Verificar que se puede guardar y cargar pickle"""
        import pickle
        import tempfile
        
        data = {'key': 'value', 'number': 42}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(data, f)
            temp_path = f.name
        
        with open(temp_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        assert loaded_data['key'] == 'value'
        assert loaded_data['number'] == 42
        
        os.unlink(temp_path)


class TestFileOperations:
    """Tests para operaciones con archivos"""
    
    def test_directory_creation(self):
        """Verificar que se pueden crear directorios"""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        test_dir = os.path.join(temp_dir, 'test_subdir')
        
        os.makedirs(test_dir, exist_ok=True)
        
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
        
        shutil.rmtree(temp_dir)
    
    def test_json_save_load(self):
        """Verificar que se puede guardar y cargar JSON"""
        import tempfile
        
        metadata = {
            'model_name': 'Random_Forest',
            'accuracy': 0.95,
            'features': ['f1', 'f2', 'f3']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(metadata, f)
            temp_path = f.name
        
        with open(temp_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['model_name'] == 'Random_Forest'
        assert len(loaded_metadata['features']) == 3
        
        os.unlink(temp_path)


class TestDataFrameOperations:
    """Tests para operaciones con DataFrames"""
    
    def test_dataframe_filtering(self):
        """Verificar filtrado de DataFrames"""
        df = pd.DataFrame({
            'type': ['TRANSFER', 'PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'amount': [100, 200, 150, 300]
        })
        
        fraud_types = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
        
        assert len(fraud_types) == 3
        assert 'PAYMENT' not in fraud_types['type'].values
    
    def test_dataframe_groupby(self):
        """Verificar agrupación de datos"""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        
        grouped = df.groupby('category')['value'].mean()
        
        assert grouped['A'] == 15
        assert grouped['B'] == 35
    
    def test_dataframe_merge(self):
        """Verificar unión de DataFrames"""
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        df2 = pd.DataFrame({
            'id': [1, 2, 3],
            'score': [90, 85, 95]
        })
        
        merged = pd.merge(df1, df2, on='id')
        
        assert len(merged) == 3
        assert 'name' in merged.columns
        assert 'score' in merged.columns


class TestNumpyOperations:
    """Tests para operaciones con NumPy"""
    
    def test_array_statistics(self):
        """Verificar cálculos estadísticos"""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        
        mean = np.mean(data)
        std = np.std(data)
        
        assert -0.2 < mean < 0.2
        assert 0.9 < std < 1.1
    
    def test_array_operations(self):
        """Verificar operaciones vectorizadas"""
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([5, 4, 3, 2, 1])
        
        result = arr1 + arr2
        
        assert len(result) == 5
        assert all(result == 6)
    
    def test_boolean_indexing(self):
        """Verificar indexación booleana"""
        arr = np.array([1, 2, 3, 4, 5, 6])
        mask = arr > 3
        
        filtered = arr[mask]
        
        assert len(filtered) == 3
        assert all(filtered > 3)


class TestConfigValidation:
    """Tests para validación de configuraciones"""
    
    def test_required_keys_present(self):
        """Verificar que las claves requeridas están presentes"""
        config = {
            'model_name': 'RandomForest',
            'n_estimators': 100,
            'random_state': 42
        }
        
        required_keys = ['model_name', 'n_estimators', 'random_state']
        
        assert all(key in config for key in required_keys)
    
    def test_value_ranges(self):
        """Verificar que los valores están en rangos válidos"""
        config = {
            'test_size': 0.2,
            'random_state': 42,
            'n_estimators': 100
        }
        
        assert 0 < config['test_size'] < 1
        assert config['n_estimators'] > 0
        assert isinstance(config['random_state'], int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
