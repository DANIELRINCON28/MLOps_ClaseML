"""
Feature Engineering Pipeline para Detección de Fraude
======================================================

Este script implementa el pipeline completo de ingeniería de características
siguiendo el patrón ColumnTransformer con pipelines especializados para cada tipo de variable.

Estructura del Pipeline:
------------------------
1. Preprocesador de Variables Numéricas (numeric_transformer)
   - SimpleImputer: Imputación de valores faltantes
   - StandardScaler/RobustScaler: Escalado de variables

2. Preprocesador de Variables Categóricas (categoric_transformer)  
   - SimpleImputer: Imputación de valores faltantes
   - OneHotEncoder/OrdinalEncoder: Codificación

3. ColumnTransformer: Combina todos los transformadores

Autores: MLOps Team
Fecha: 2025
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class FraudFeatureEngineering:
    """
    Clase para manejar toda la ingeniería de características 
    del proyecto de detección de fraude.
    """
    
    def __init__(self, data_path='../../data/processed/df_original.pkl'):
        """
        Inicializa el ingeniero de características.
        
        Parameters:
        -----------
        data_path : str
            Ruta al archivo de datos
        """
        self.data_path = data_path
        self.df = None
        self.df_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        
        print("=" * 80)
        print("FRAUD DETECTION - FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
    
    
    def load_data(self):
        """Carga los datos desde el archivo pickle."""
        print("\n🔄 Cargando datos...")
        
        try:
            self.df = pd.read_pickle(self.data_path)
            print(f"✅ Datos cargados: {self.df.shape[0]:,} filas x {self.df.shape[1]} columnas")
        except:
            # Intentar desde CSV si pickle no existe
            csv_path = '../../Base_datos.csv'
            self.df = pd.read_csv(csv_path)
            print(f"✅ Datos cargados desde CSV: {self.df.shape[0]:,} filas x {self.df.shape[1]} columnas")
        
        return self.df
    
    
    def create_features(self):
        """
        Crea nuevas características derivadas basadas en el análisis exploratorio.
        """
        print("\n🔧 Creando nuevas características...")
        
        self.df_features = self.df.copy()
        
        # 1. FEATURES DE DIFERENCIA DE BALANCES
        print("  📊 Creando features de balance...")
        
        self.df_features['balance_diff_orig'] = (
            self.df_features['oldbalanceOrg'] - self.df_features['newbalanceOrig']
        )
        
        self.df_features['balance_diff_dest'] = (
            self.df_features['newbalanceDest'] - self.df_features['oldbalanceDest']
        )
        
        self.df_features['error_balance_orig'] = np.abs(
            self.df_features['balance_diff_orig'] - self.df_features['amount']
        )
        
        self.df_features['error_balance_dest'] = np.abs(
            self.df_features['balance_diff_dest'] - self.df_features['amount']
        )
        
        self.df_features['error_balance_total'] = (
            self.df_features['error_balance_orig'] + self.df_features['error_balance_dest']
        )
        
        
        # 2. FEATURES BINARIOS
        print("  📊 Creando features binarios...")
        
        self.df_features['orig_is_merchant'] = (
            self.df_features['nameOrig'].str[0] == 'M'
        ).astype(int)
        
        self.df_features['dest_is_merchant'] = (
            self.df_features['nameDest'].str[0] == 'M'
        ).astype(int)
        
        self.df_features['orig_balance_zero_after'] = (
            self.df_features['newbalanceOrig'] == 0
        ).astype(int)
        
        self.df_features['dest_balance_zero_after'] = (
            self.df_features['newbalanceDest'] == 0
        ).astype(int)
        
        self.df_features['orig_balance_zero_before'] = (
            self.df_features['oldbalanceOrg'] == 0
        ).astype(int)
        
        self.df_features['dest_balance_zero_before'] = (
            self.df_features['oldbalanceDest'] == 0
        ).astype(int)
        
        
        # 3. FEATURES DE RATIOS
        print("  📊 Creando features de ratios...")
        
        self.df_features['amount_to_oldbalance_orig_ratio'] = (
            self.df_features['amount'] / (self.df_features['oldbalanceOrg'] + 1)
        )
        
        self.df_features['amount_to_oldbalance_dest_ratio'] = (
            self.df_features['amount'] / (self.df_features['oldbalanceDest'] + 1)
        )
        
        self.df_features['balance_ratio_orig'] = (
            self.df_features['newbalanceOrig'] / (self.df_features['oldbalanceOrg'] + 1)
        )
        
        self.df_features['balance_ratio_dest'] = (
            self.df_features['newbalanceDest'] / (self.df_features['oldbalanceDest'] + 1)
        )
        
        
        # 4. FEATURES TEMPORALES
        print("  📊 Creando features temporales...")
        
        self.df_features['hour_of_day'] = self.df_features['step'] % 24
        self.df_features['day_of_month'] = (self.df_features['step'] // 24) + 1
        
        self.df_features['is_weekend'] = (
            ((self.df_features['step'] // 24) % 7) >= 5
        ).astype(int)
        
        self.df_features['is_night'] = (
            (self.df_features['hour_of_day'] >= 22) | 
            (self.df_features['hour_of_day'] <= 6)
        ).astype(int)
        
        
        # 5. FEATURES DE TIPO DE TRANSACCIÓN
        print("  📊 Creando features de tipo de transacción...")
        
        fraud_types = ['TRANSFER', 'CASH_OUT']
        self.df_features['is_fraud_type'] = (
            self.df_features['type'].isin(fraud_types)
        ).astype(int)
        
        
        # 6. FEATURES DE MAGNITUD
        print("  📊 Creando features de magnitud...")
        
        self.df_features['is_large_transaction'] = (
            self.df_features['amount'] > 200000
        ).astype(int)
        
        self.df_features['amount_category'] = pd.cut(
            self.df_features['amount'],
            bins=[0, 1000, 10000, 100000, float('inf')],
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        
        print(f"✅ {len(self.df_features.columns) - len(self.df.columns)} nuevas features creadas")
        print(f"📊 Total de columnas: {len(self.df_features.columns)}")
        
        return self.df_features
    
    
    def prepare_for_modeling(self, test_size=0.2, random_state=42):
        """
        Prepara los datos para el modelado.
        """
        print("\n📦 Preparando datos para modelado...")
        
        columns_to_drop = [
            'nameOrig',
            'nameDest',
            'isFraud',
            'isFlaggedFraud'
        ]
        
        X = self.df_features.drop(columns=columns_to_drop)
        y = self.df_features['isFraud']
        
        print(f"  📊 Features (X): {X.shape}")
        print(f"  🎯 Target (y): {y.shape}")
        print(f"  📊 Distribución de clases:")
        print(f"     - No Fraude: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
        print(f"     - Fraude: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        print(f"\n✅ División completada:")
        print(f"  📊 Train: {self.X_train.shape[0]:,} muestras")
        print(f"  📊 Test: {self.X_test.shape[0]:,} muestras")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    
    def build_preprocessor(self):
        """
        Construye el pipeline de preprocesamiento usando ColumnTransformer.
        """
        print("\n🏗️ Construyendo pipeline de preprocesamiento...")
        
        numeric_features = self.X_train.select_dtypes(
            include=['int8', 'int16', 'int32', 'int64', 'float32', 'float64']
        ).columns.tolist()
        
        categorical_features = self.X_train.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        print(f"  📊 Variables numéricas: {len(numeric_features)}")
        print(f"  📊 Variables categóricas: {len(categorical_features)}")
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        categoric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categoric_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        print("✅ Pipeline de preprocesamiento construido")
        
        return self.preprocessor
    
    
    def fit_transform_data(self):
        """
        Ajusta el preprocesador y transforma los datos.
        """
        print("\n🔄 Ajustando y transformando datos...")
        
        X_train_processed = self.preprocessor.fit_transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"✅ Datos transformados")
        print(f"  📊 X_train procesado: {X_train_processed.shape}")
        print(f"  📊 X_test procesado: {X_test_processed.shape}")
        
        feature_names = self._get_feature_names()
        
        X_train_processed_df = pd.DataFrame(
            X_train_processed, 
            columns=feature_names,
            index=self.X_train.index
        )
        
        X_test_processed_df = pd.DataFrame(
            X_test_processed, 
            columns=feature_names,
            index=self.X_test.index
        )
        
        return X_train_processed_df, X_test_processed_df
    
    
    def _get_feature_names(self):
        """Obtiene los nombres de las features después de la transformación."""
        feature_names = []
        
        num_features = self.preprocessor.transformers_[0][2]
        feature_names.extend(num_features)
        
        if len(self.preprocessor.transformers_[1][2]) > 0:
            cat_encoder = self.preprocessor.transformers_[1][1].named_steps['encoder']
            cat_features = cat_encoder.get_feature_names_out(
                self.preprocessor.transformers_[1][2]
            )
            feature_names.extend(cat_features)
        
        return feature_names
    
    
    def save_artifacts(self, output_dir='../../data/processed'):
        """
        Guarda todos los artefactos.
        """
        print("\n💾 Guardando artefactos...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.X_train.to_pickle(f'{output_dir}/X_train.pkl')
        self.X_test.to_pickle(f'{output_dir}/X_test.pkl')
        self.y_train.to_pickle(f'{output_dir}/y_train.pkl')
        self.y_test.to_pickle(f'{output_dir}/y_test.pkl')
        
        print(f"  ✅ Datasets guardados")
        
        with open(f'{output_dir}/preprocessor.pkl', 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"  ✅ Preprocesador guardado")
        
        self.df_features.to_pickle(f'{output_dir}/df_features_complete.pkl')
        print(f"  ✅ Dataset completo guardado")
        
        metadata = {
            'n_features': self.X_train.shape[1],
            'n_samples_train': self.X_train.shape[0],
            'n_samples_test': self.X_test.shape[0],
            'feature_names': list(self.X_train.columns),
            'class_distribution_train': {
                'no_fraud': int((self.y_train == 0).sum()),
                'fraud': int((self.y_train == 1).sum())
            },
            'class_distribution_test': {
                'no_fraud': int((self.y_test == 0).sum()),
                'fraud': int((self.y_test == 1).sum())
            }
        }
        
        with open(f'{output_dir}/feature_engineering_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  ✅ Metadatos guardados")
        
        return metadata


def main():
    """Función principal."""
    
    fe = FraudFeatureEngineering()
    fe.load_data()
    fe.create_features()
    fe.prepare_for_modeling(test_size=0.2, random_state=42)
    fe.build_preprocessor()
    X_train_processed, X_test_processed = fe.fit_transform_data()
    metadata = fe.save_artifacts()
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETADO ✅")
    print("=" * 80)
    print(f"\n📊 Total de features: {metadata['n_features']}")
    print(f"📊 Muestras de entrenamiento: {metadata['n_samples_train']:,}")
    print(f"📊 Muestras de prueba: {metadata['n_samples_test']:,}")
    print("\n➡️ Siguiente paso: Model Training")
    print("=" * 80)
    
    return fe, X_train_processed, X_test_processed


if __name__ == "__main__":
    fe, X_train_processed, X_test_processed = main()