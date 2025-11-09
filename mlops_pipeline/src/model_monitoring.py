"""
Sistema de Monitoreo y Detección de Data Drift
Pipeline MLOps - Detección de Fraude

Este módulo implementa un sistema completo de monitoreo que:
1. Carga datos históricos (entrenamiento) y nuevos datos (producción)
2. Genera predicciones con el modelo entrenado
3. Calcula métricas de data drift (KS, PSI, JS, Chi-cuadrado)
4. Detecta cambios en la distribución de variables
5. Genera alertas automáticas
6. Guarda resultados para visualización en Streamlit
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Obtener directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class DataDriftMonitor:
    """
    Clase para monitorear data drift en datos de producción
    """
    
    def __init__(self, reference_data_path, model_path, preprocessor_path):
        """
        Inicializa el monitor de data drift
        
        Args:
            reference_data_path: Ruta a los datos de referencia (entrenamiento)
            model_path: Ruta al modelo entrenado
            preprocessor_path: Ruta al preprocesador
        """
        self.reference_data_path = Path(reference_data_path)
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        
        # Umbrales para alertas
        self.thresholds = {
            'ks_stat': 0.1,      # Kolmogorov-Smirnov
            'psi': 0.2,          # Population Stability Index
            'js_divergence': 0.1, # Jensen-Shannon
            'chi2_pvalue': 0.05   # Chi-cuadrado
        }
        
        # Cargar datos de referencia y modelo
        self.load_reference_data()
        self.load_model()
        
    def load_reference_data(self):
        """Carga los datos de referencia (entrenamiento)"""
        print("📂 Cargando datos de referencia...")
        
        # Cargar datos procesados de entrenamiento usando rutas absolutas
        data_dir = PROJECT_ROOT / 'data' / 'processed'
        
        with open(data_dir / 'X_train.pkl', 'rb') as f:
            self.X_reference = pickle.load(f)
        
        with open(data_dir / 'y_train.pkl', 'rb') as f:
            self.y_reference = pickle.load(f)
        
        # Cargar datos originales para comparación
        with open(data_dir / 'df_features_complete.pkl', 'rb') as f:
            self.df_reference = pickle.load(f)
        
        print(f"✅ Datos de referencia cargados: {self.X_reference.shape}")
        
    def load_model(self):
        """Carga el modelo y preprocesador entrenados"""
        print("🤖 Cargando modelo entrenado...")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        print("✅ Modelo y preprocesador cargados")
        
    def load_production_data(self, production_data_path):
        """
        Carga datos de producción para monitoreo
        
        Args:
            production_data_path: Ruta a los datos de producción
        """
        print(f"📊 Cargando datos de producción: {production_data_path}")
        
        # Cargar datos de producción (puede ser CSV o pickle)
        if str(production_data_path).endswith('.pkl'):
            with open(production_data_path, 'rb') as f:
                self.df_production = pickle.load(f)
        else:
            self.df_production = pd.read_csv(production_data_path)
        
        print(f"✅ Datos de producción cargados: {self.df_production.shape}")
        
        return self.df_production
    
    def preprocess_production_data(self):
        """Preprocesa los datos de producción usando el preprocesador entrenado"""
        print("🔧 Preprocesando datos de producción...")
        
        # Aplicar el mismo preprocesamiento que en entrenamiento
        # (Aquí asumimos que df_production tiene las mismas columnas que los datos de entrenamiento)
        
        # Separar features y target (si existe)
        if 'isFraud' in self.df_production.columns:
            X_prod = self.df_production.drop('isFraud', axis=1)
            y_prod = self.df_production['isFraud']
        else:
            X_prod = self.df_production.copy()
            y_prod = None
        
        # Aplicar preprocesador
        X_prod_processed = self.preprocessor.transform(X_prod)
        
        self.X_production = X_prod_processed
        self.y_production = y_prod
        
        print(f"✅ Datos preprocesados: {self.X_production.shape}")
        
        return self.X_production, self.y_production
    
    def generate_predictions(self):
        """Genera predicciones para los datos de producción"""
        print("🎯 Generando predicciones...")
        
        # Predicciones binarias
        self.predictions = self.model.predict(self.X_production)
        
        # Probabilidades
        self.prediction_proba = self.model.predict_proba(self.X_production)[:, 1]
        
        print(f"✅ Predicciones generadas: {len(self.predictions)}")
        print(f"   - Fraudes detectados: {self.predictions.sum()} ({self.predictions.sum()/len(self.predictions)*100:.2f}%)")
        
        return self.predictions, self.prediction_proba
    
    def calculate_ks_statistic(self, reference_col, production_col, col_name):
        """
        Calcula Kolmogorov-Smirnov test
        
        Args:
            reference_col: Columna de datos de referencia
            production_col: Columna de datos de producción
            col_name: Nombre de la columna
        
        Returns:
            dict con estadísticos KS
        """
        # Remover NaN
        ref_clean = reference_col.dropna()
        prod_clean = production_col.dropna()
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(ref_clean, prod_clean)
        
        # Determinar severidad
        if ks_stat < self.thresholds['ks_stat']:
            severity = 'low'
            status = '✅'
        elif ks_stat < self.thresholds['ks_stat'] * 2:
            severity = 'medium'
            status = '⚠️'
        else:
            severity = 'high'
            status = '🚨'
        
        return {
            'variable': col_name,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'severity': severity,
            'status': status,
            'drift_detected': ks_stat >= self.thresholds['ks_stat']
        }
    
    def calculate_psi(self, reference_col, production_col, col_name, bins=10):
        """
        Calcula Population Stability Index (PSI)
        
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        
        Interpretación:
        - PSI < 0.1: Sin cambio significativo
        - 0.1 <= PSI < 0.2: Cambio moderado
        - PSI >= 0.2: Cambio significativo
        """
        # Remover NaN
        ref_clean = reference_col.dropna()
        prod_clean = production_col.dropna()
        
        # Crear bins basados en datos de referencia
        min_val = ref_clean.min()
        max_val = ref_clean.max()
        
        breakpoints = np.linspace(min_val, max_val, bins + 1)
        
        # Calcular distribuciones
        ref_counts, _ = np.histogram(ref_clean, bins=breakpoints)
        prod_counts, _ = np.histogram(prod_clean, bins=breakpoints)
        
        # Convertir a porcentajes
        ref_percents = ref_counts / len(ref_clean)
        prod_percents = prod_counts / len(prod_clean)
        
        # Evitar división por cero
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        prod_percents = np.where(prod_percents == 0, 0.0001, prod_percents)
        
        # Calcular PSI
        psi_values = (prod_percents - ref_percents) * np.log(prod_percents / ref_percents)
        psi = np.sum(psi_values)
        
        # Determinar severidad
        if psi < 0.1:
            severity = 'low'
            status = '✅'
        elif psi < 0.2:
            severity = 'medium'
            status = '⚠️'
        else:
            severity = 'high'
            status = '🚨'
        
        return {
            'variable': col_name,
            'psi': psi,
            'severity': severity,
            'status': status,
            'drift_detected': psi >= self.thresholds['psi']
        }
    
    def calculate_js_divergence(self, reference_col, production_col, col_name, bins=10):
        """
        Calcula Jensen-Shannon divergence
        
        JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        donde M = 0.5 * (P + Q)
        """
        # Remover NaN
        ref_clean = reference_col.dropna()
        prod_clean = production_col.dropna()
        
        # Crear bins
        min_val = min(ref_clean.min(), prod_clean.min())
        max_val = max(ref_clean.max(), prod_clean.max())
        
        breakpoints = np.linspace(min_val, max_val, bins + 1)
        
        # Calcular distribuciones
        ref_counts, _ = np.histogram(ref_clean, bins=breakpoints)
        prod_counts, _ = np.histogram(prod_clean, bins=breakpoints)
        
        # Normalizar
        ref_dist = ref_counts / ref_counts.sum()
        prod_dist = prod_counts / prod_counts.sum()
        
        # Evitar ceros
        ref_dist = np.where(ref_dist == 0, 1e-10, ref_dist)
        prod_dist = np.where(prod_dist == 0, 1e-10, prod_dist)
        
        # Calcular JS divergence
        js_div = jensenshannon(ref_dist, prod_dist)
        
        # Determinar severidad
        if js_div < self.thresholds['js_divergence']:
            severity = 'low'
            status = '✅'
        elif js_div < self.thresholds['js_divergence'] * 2:
            severity = 'medium'
            status = '⚠️'
        else:
            severity = 'high'
            status = '🚨'
        
        return {
            'variable': col_name,
            'js_divergence': js_div,
            'severity': severity,
            'status': status,
            'drift_detected': js_div >= self.thresholds['js_divergence']
        }
    
    def calculate_chi2_test(self, reference_col, production_col, col_name):
        """
        Calcula Chi-cuadrado test para variables categóricas
        """
        # Obtener categorías únicas
        all_categories = set(reference_col.unique()) | set(production_col.unique())
        
        # Contar frecuencias
        ref_counts = reference_col.value_counts().reindex(all_categories, fill_value=0)
        prod_counts = production_col.value_counts().reindex(all_categories, fill_value=0)
        
        # Crear tabla de contingencia
        contingency_table = np.array([ref_counts, prod_counts])
        
        # Chi-cuadrado test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Determinar severidad
        if p_value >= self.thresholds['chi2_pvalue']:
            severity = 'low'
            status = '✅'
        elif p_value >= self.thresholds['chi2_pvalue'] / 2:
            severity = 'medium'
            status = '⚠️'
        else:
            severity = 'high'
            status = '🚨'
        
        return {
            'variable': col_name,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'severity': severity,
            'status': status,
            'drift_detected': p_value < self.thresholds['chi2_pvalue']
        }
    
    def detect_drift(self, sample_size=None):
        """
        Detecta data drift comparando datos de referencia vs producción
        
        Args:
            sample_size: Tamaño de muestra para análisis (None = todos los datos)
        """
        print("\n" + "="*80)
        print("🔍 INICIANDO DETECCIÓN DE DATA DRIFT")
        print("="*80)
        
        drift_results = []
        
        # Muestreo si se especifica
        if sample_size:
            df_ref_sample = self.df_reference.sample(min(sample_size, len(self.df_reference)), random_state=42)
            df_prod_sample = self.df_production.sample(min(sample_size, len(self.df_production)), random_state=42)
        else:
            df_ref_sample = self.df_reference
            df_prod_sample = self.df_production
        
        # Seleccionar columnas numéricas para análisis
        numeric_columns = df_ref_sample.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col in df_prod_sample.columns and col != 'isFraud']
        
        print(f"\n📊 Analizando {len(numeric_columns)} variables numéricas...")
        
        for col in numeric_columns:
            if col in df_prod_sample.columns:
                print(f"\n   Analizando: {col}")
                
                ref_col = df_ref_sample[col]
                prod_col = df_prod_sample[col]
                
                # Calcular métricas de drift
                ks_result = self.calculate_ks_statistic(ref_col, prod_col, col)
                psi_result = self.calculate_psi(ref_col, prod_col, col)
                js_result = self.calculate_js_divergence(ref_col, prod_col, col)
                
                # Combinar resultados
                drift_info = {
                    'variable': col,
                    'tipo': 'numérica',
                    'ks_statistic': ks_result['ks_statistic'],
                    'ks_p_value': ks_result['p_value'],
                    'psi': psi_result['psi'],
                    'js_divergence': js_result['js_divergence'],
                    'drift_detected': (ks_result['drift_detected'] or 
                                     psi_result['drift_detected'] or 
                                     js_result['drift_detected']),
                    'severity': max([ks_result['severity'], psi_result['severity'], js_result['severity']],
                                  key=lambda x: {'low': 0, 'medium': 1, 'high': 2}[x]),
                    'ref_mean': float(ref_col.mean()),
                    'prod_mean': float(prod_col.mean()),
                    'ref_std': float(ref_col.std()),
                    'prod_std': float(prod_col.std()),
                    'mean_change_%': float((prod_col.mean() - ref_col.mean()) / ref_col.mean() * 100) if ref_col.mean() != 0 else 0
                }
                
                drift_results.append(drift_info)
                
                # Mostrar estado
                status = '🚨' if drift_info['severity'] == 'high' else '⚠️' if drift_info['severity'] == 'medium' else '✅'
                print(f"      {status} KS={ks_result['ks_statistic']:.4f}, PSI={psi_result['psi']:.4f}, JS={js_result['js_divergence']:.4f}")
        
        # Analizar variables categóricas
        categorical_columns = df_ref_sample.select_dtypes(include=['object', 'category']).columns
        categorical_columns = [col for col in categorical_columns if col in df_prod_sample.columns]
        
        if len(categorical_columns) > 0:
            print(f"\n📊 Analizando {len(categorical_columns)} variables categóricas...")
            
            for col in categorical_columns:
                print(f"\n   Analizando: {col}")
                
                ref_col = df_ref_sample[col]
                prod_col = df_prod_sample[col]
                
                chi2_result = self.calculate_chi2_test(ref_col, prod_col, col)
                
                drift_info = {
                    'variable': col,
                    'tipo': 'categórica',
                    'chi2_statistic': chi2_result['chi2_statistic'],
                    'chi2_p_value': chi2_result['p_value'],
                    'drift_detected': chi2_result['drift_detected'],
                    'severity': chi2_result['severity'],
                    'ref_categories': len(ref_col.unique()),
                    'prod_categories': len(prod_col.unique())
                }
                
                drift_results.append(drift_info)
                
                status = '🚨' if drift_info['severity'] == 'high' else '⚠️' if drift_info['severity'] == 'medium' else '✅'
                print(f"      {status} Chi2={chi2_result['chi2_statistic']:.4f}, p-value={chi2_result['p_value']:.4f}")
        
        self.drift_results = pd.DataFrame(drift_results)
        
        return self.drift_results
    
    def generate_alerts(self):
        """Genera alertas basadas en los resultados de drift"""
        print("\n" + "="*80)
        print("🚨 GENERACIÓN DE ALERTAS")
        print("="*80)
        
        alerts = []
        
        # Alertas por severidad alta
        high_severity = self.drift_results[self.drift_results['severity'] == 'high']
        
        if len(high_severity) > 0:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'level': 'CRÍTICO',
                'message': f'🚨 ALERTA CRÍTICA: {len(high_severity)} variables con drift severo detectado',
                'variables': high_severity['variable'].tolist(),
                'recommendation': 'ACCIÓN INMEDIATA REQUERIDA: Considerar reentrenamiento del modelo',
                'details': high_severity.to_dict('records')
            }
            alerts.append(alert)
            print(f"\n🚨 CRÍTICO: {len(high_severity)} variables con drift severo")
            print(f"   Variables: {', '.join(high_severity['variable'].tolist())}")
        
        # Alertas por severidad media
        medium_severity = self.drift_results[self.drift_results['severity'] == 'medium']
        
        if len(medium_severity) > 0:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'level': 'ADVERTENCIA',
                'message': f'⚠️ ADVERTENCIA: {len(medium_severity)} variables con drift moderado',
                'variables': medium_severity['variable'].tolist(),
                'recommendation': 'Monitorear de cerca estas variables en los próximos períodos',
                'details': medium_severity.to_dict('records')
            }
            alerts.append(alert)
            print(f"\n⚠️ ADVERTENCIA: {len(medium_severity)} variables con drift moderado")
            print(f"   Variables: {', '.join(medium_severity['variable'].tolist())}")
        
        # Resumen general
        drift_detected = self.drift_results[self.drift_results['drift_detected'] == True]
        
        summary_alert = {
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'message': f'📊 Resumen: {len(drift_detected)}/{len(self.drift_results)} variables con drift detectado',
            'total_variables': len(self.drift_results),
            'drift_detected': len(drift_detected),
            'high_severity': len(high_severity),
            'medium_severity': len(medium_severity),
            'recommendation': 'Revisar dashboard de monitoreo para más detalles'
        }
        alerts.append(summary_alert)
        
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   Total variables analizadas: {len(self.drift_results)}")
        print(f"   Variables con drift: {len(drift_detected)}")
        print(f"   Severidad alta: {len(high_severity)}")
        print(f"   Severidad media: {len(medium_severity)}")
        
        self.alerts = alerts
        
        return alerts
    
    def save_results(self, output_dir='../../outputs/monitoring'):
        """Guarda los resultados del monitoreo"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\n💾 Guardando resultados en {output_path}...")
        
        # Guardar resultados de drift
        drift_file = output_path / f'drift_results_{timestamp}.csv'
        self.drift_results.to_csv(drift_file, index=False)
        print(f"   ✅ Drift results: {drift_file}")
        
        # Guardar alertas
        alerts_file = output_path / f'alerts_{timestamp}.json'
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump(self.alerts, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Alerts: {alerts_file}")
        
        # Guardar predicciones
        predictions_df = pd.DataFrame({
            'prediction': self.predictions,
            'prediction_proba': self.prediction_proba
        })
        
        # Agregar datos originales si están disponibles
        if hasattr(self, 'df_production'):
            predictions_df = pd.concat([self.df_production.reset_index(drop=True), predictions_df], axis=1)
        
        predictions_file = output_path / f'predictions_{timestamp}.csv'
        predictions_df.to_csv(predictions_file, index=False)
        print(f"   ✅ Predictions: {predictions_file}")
        
        # Guardar resumen para Streamlit
        summary = {
            'timestamp': timestamp,
            'total_variables': len(self.drift_results),
            'drift_detected': int(self.drift_results['drift_detected'].sum()),
            'high_severity': int((self.drift_results['severity'] == 'high').sum()),
            'medium_severity': int((self.drift_results['severity'] == 'medium').sum()),
            'low_severity': int((self.drift_results['severity'] == 'low').sum()),
            'predictions': {
                'total': len(self.predictions),
                'fraud_detected': int(self.predictions.sum()),
                'fraud_rate': float(self.predictions.sum() / len(self.predictions) * 100)
            }
        }
        
        summary_file = output_path / 'latest_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✅ Summary: {summary_file}")
        
        print("\n✅ Todos los resultados guardados exitosamente")
        
        return {
            'drift_results': drift_file,
            'alerts': alerts_file,
            'predictions': predictions_file,
            'summary': summary_file
        }


def main():
    """Función principal para ejecutar el monitoreo"""
    print("\n" + "="*80)
    print("🔍 SISTEMA DE MONITOREO Y DETECCIÓN DE DATA DRIFT")
    print("Pipeline MLOps - Detección de Fraude")
    print("="*80 + "\n")
    
    # Rutas a los archivos
    model_path = '../../models/xgboost_model.pkl'
    preprocessor_path = '../../data/processed/preprocessor.pkl'
    reference_data_path = '../../data/processed/df_features_complete.pkl'
    
    # Crear monitor
    monitor = DataDriftMonitor(
        reference_data_path=reference_data_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )
    
    # Cargar datos de producción (simulación usando datos de test)
    # En producción real, esto vendría de una base de datos o API
    print("\n📊 Simulando datos de producción...")
    print("   (En producción, estos datos vendrían de la base de datos en tiempo real)")
    
    # Usar los datos de test como "producción"
    production_data = monitor.load_production_data('../../data/processed/df_features_complete.pkl')
    
    # Tomar una muestra para simular datos nuevos
    production_sample = production_data.sample(frac=0.3, random_state=42)
    
    # Simular algunos cambios en los datos (para demostrar drift)
    print("\n⚡ Simulando cambios en la distribución de datos...")
    production_sample_modified = production_sample.copy()
    
    # Modificar algunas columnas para inducir drift
    if 'amount' in production_sample_modified.columns:
        production_sample_modified['amount'] = production_sample_modified['amount'] * 1.2  # 20% más
    
    if 'oldbalanceOrg' in production_sample_modified.columns:
        production_sample_modified['oldbalanceOrg'] = production_sample_modified['oldbalanceOrg'] * 0.8  # 20% menos
    
    # Guardar temporalmente
    temp_prod_path = '../../data/processed/temp_production_data.csv'
    production_sample_modified.to_csv(temp_prod_path, index=False)
    
    # Cargar datos de producción modificados
    monitor.df_production = production_sample_modified
    
    # Preprocesar datos de producción
    monitor.preprocess_production_data()
    
    # Generar predicciones
    monitor.generate_predictions()
    
    # Detectar drift (muestreo de 5000 registros para análisis más rápido)
    drift_results = monitor.detect_drift(sample_size=5000)
    
    # Generar alertas
    alerts = monitor.generate_alerts()
    
    # Guardar resultados
    saved_files = monitor.save_results()
    
    print("\n" + "="*80)
    print("✅ MONITOREO COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("\n📊 Los resultados están disponibles para visualización en Streamlit")
    print("   Ejecuta: streamlit run app_monitoring.py")
    print("\n")


if __name__ == "__main__":
    main()
