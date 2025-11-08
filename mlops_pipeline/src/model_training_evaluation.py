"""
Model Training and Evaluation Pipeline para Detección de Fraude
================================================================

Este script entrena y evalúa múltiples modelos de Machine Learning
para seleccionar el mejor basado en performance, consistency y scalability.

Modelos a evaluar:
------------------
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. LightGBM Classifier
5. Gradient Boosting Classifier

Métricas de evaluación:
-----------------------
- ROC-AUC Score
- Precision-Recall AUC
- F1-Score
- Precision
- Recall
- Accuracy

Autores: MLOps Team
Fecha: 2025
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Obtener directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Métricas y evaluación
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# Balanceo de clases
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')


class ModelTrainingEvaluation:
    """
    Clase para entrenar y evaluar múltiples modelos de ML.
    """
    
    def __init__(self, data_dir=None):
        """
        Inicializa el entrenador de modelos.
        
        Parameters:
        -----------
        data_dir : str
            Directorio donde están los datos procesados
        """
        if data_dir is None:
            data_dir = PROJECT_ROOT / 'data' / 'processed'
        self.data_dir = data_dir
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        print("=" * 80)
        print("FRAUD DETECTION - MODEL TRAINING & EVALUATION")
        print("=" * 80)
    
    
    def load_data(self):
        """Carga los datos preprocesados."""
        print("\n🔄 Cargando datos preprocesados...")
        
        self.X_train = pd.read_pickle(f'{self.data_dir}/X_train.pkl')
        self.X_test = pd.read_pickle(f'{self.data_dir}/X_test.pkl')
        self.y_train = pd.read_pickle(f'{self.data_dir}/y_train.pkl')
        self.y_test = pd.read_pickle(f'{self.data_dir}/y_test.pkl')
        
        print(f"✅ Datos cargados")
        print(f"  📊 X_train: {self.X_train.shape}")
        print(f"  📊 X_test: {self.X_test.shape}")
        print(f"  🎯 y_train: {self.y_train.shape} - Fraude: {self.y_train.sum():,}")
        print(f"  🎯 y_test: {self.y_test.shape} - Fraude: {self.y_test.sum():,}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    
    def apply_smote(self, sampling_strategy=0.3):
        """
        Aplica SMOTE para balancear las clases en el conjunto de entrenamiento.
        
        Parameters:
        -----------
        sampling_strategy : float
            Ratio de la clase minoritaria vs mayoritaria después del rebalanceo
        """
        print(f"\n⚖️ Aplicando SMOTE (sampling_strategy={sampling_strategy})...")
        
        print(f"  Antes - Clase 0: {(self.y_train==0).sum():,}, Clase 1: {(self.y_train==1).sum():,}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
        
        self.X_train = X_train_resampled
        self.y_train = y_train_resampled
        
        print(f"  Después - Clase 0: {(self.y_train==0).sum():,}, Clase 1: {(self.y_train==1).sum():,}")
        print(f"  ✅ SMOTE aplicado")
        
        return self.X_train, self.y_train
    
    
    def define_models(self):
        """
        Define los modelos a entrenar con sus hiperparámetros iniciales.
        """
        print("\n🤖 Definiendo modelos...")
        
        self.models = {
            'Logistic_Regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'description': 'Modelo lineal simple y interpretable'
            },
            
            'Random_Forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'description': 'Ensemble de árboles de decisión'
            },
            
            'XGBoost': {
                'model': XGBClassifier(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=(self.y_train==0).sum()/(self.y_train==1).sum(),
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                ),
                'description': 'Gradient Boosting optimizado'
            },
            
            'LightGBM': {
                'model': LGBMClassifier(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'description': 'Gradient Boosting ligero y rápido'
            },
            
            'Gradient_Boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
                ),
                'description': 'Gradient Boosting de sklearn'
            }
        }
        
        print(f"✅ {len(self.models)} modelos definidos:")
        for name, info in self.models.items():
            print(f"  • {name}: {info['description']}")
        
        return self.models
    
    
    def train_models(self):
        """
        Entrena todos los modelos definidos.
        """
        print("\n🏋️ Entrenando modelos...")
        print("=" * 80)
        
        for name, model_info in self.models.items():
            print(f"\n🔄 Entrenando {name}...")
            start_time = datetime.now()
            
            model = model_info['model']
            model.fit(self.X_train, self.y_train)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Guardar modelo entrenado y tiempo
            self.models[name]['trained_model'] = model
            self.models[name]['training_time'] = training_time
            
            print(f"  ✅ {name} entrenado en {training_time:.2f} segundos")
        
        print("\n" + "=" * 80)
        print("✅ Todos los modelos entrenados")
        
        return self.models
    
    
    def evaluate_models(self):
        """
        Evalúa todos los modelos entrenados en el conjunto de prueba.
        """
        print("\n📊 Evaluando modelos...")
        print("=" * 80)
        
        for name, model_info in self.models.items():
            print(f"\n📈 Evaluando {name}...")
            
            model = model_info['trained_model']
            
            # Predicciones
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Métricas
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'pr_auc': average_precision_score(self.y_test, y_pred_proba)
            }
            
            # Guardar resultados
            self.results[name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred),
                'training_time': model_info['training_time']
            }
            
            # Imprimir métricas
            print(f"\n  Métricas de {name}:")
            print(f"  {'Métrica':<20} {'Valor':<10}")
            print(f"  {'-'*30}")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name:<20} {metric_value:.4f}")
            print(f"  {'Tiempo entrenamiento':<20} {model_info['training_time']:.2f}s")
        
        print("\n" + "=" * 80)
        print("✅ Evaluación completada")
        
        return self.results
    
    
    def compare_models(self):
        """
        Compara todos los modelos y crea visualizaciones.
        """
        print("\n📊 Comparando modelos...")
        
        # Crear DataFrame con métricas
        comparison_data = []
        for name, results in self.results.items():
            row = {'Model': name}
            row.update(results['metrics'])
            row['Training_Time'] = results['training_time']
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('roc_auc', ascending=False)
        
        print("\n📋 TABLA DE COMPARACIÓN DE MODELOS:")
        print("=" * 80)
        display(df_comparison)
        
        # Guardar tabla
        df_comparison.to_csv('../../outputs/model_comparison.csv', index=False)
        print("\n✅ Tabla guardada en outputs/model_comparison.csv")
        
        # Visualizaciones
        self._plot_metrics_comparison(df_comparison)
        self._plot_roc_curves()
        self._plot_precision_recall_curves()
        self._plot_confusion_matrices()
        
        return df_comparison
    
    
    def _plot_metrics_comparison(self, df_comparison):
        """Gráfico de barras comparando métricas."""
        print("\n📊 Generando gráfico de comparación de métricas...")
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics_to_plot):
            df_sorted = df_comparison.sort_values(metric, ascending=False)
            
            axes[idx].barh(df_sorted['Model'], df_sorted[metric], color='steelblue', alpha=0.7)
            axes[idx].set_xlabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'{metric.replace("_", " ").title()} por Modelo', fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
            
            # Añadir valores
            for i, v in enumerate(df_sorted[metric]):
                axes[idx].text(v, i, f' {v:.4f}', va='center')
        
        plt.tight_layout()
        plt.savefig('../../outputs/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Gráfico guardado en outputs/metrics_comparison.png")
    
    
    def _plot_roc_curves(self):
        """Gráficos de curvas ROC."""
        print("\n📊 Generando curvas ROC...")
        
        plt.figure(figsize=(12, 8))
        
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            auc_score = results['metrics']['roc_auc']
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Comparación de Modelos', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../outputs/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Curvas ROC guardadas en outputs/roc_curves.png")
    
    
    def _plot_precision_recall_curves(self):
        """Gráficos de curvas Precision-Recall."""
        print("\n📊 Generando curvas Precision-Recall...")
        
        plt.figure(figsize=(12, 8))
        
        for name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, results['y_pred_proba'])
            pr_auc = results['metrics']['pr_auc']
            
            plt.plot(recall, precision, linewidth=2, label=f'{name} (AUC = {pr_auc:.4f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Comparación de Modelos', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../../outputs/pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Curvas PR guardadas en outputs/pr_curves.png")
    
    
    def _plot_confusion_matrices(self):
        """Matrices de confusión para cada modelo."""
        print("\n📊 Generando matrices de confusión...")
        
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.ravel()
        
        for idx, (name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{name}', fontweight='bold')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_xticklabels(['No Fraud', 'Fraud'])
            axes[idx].set_yticklabels(['No Fraud', 'Fraud'])
        
        # Ocultar axes sobrantes
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('../../outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Matrices de confusión guardadas en outputs/confusion_matrices.png")
    
    
    def select_best_model(self, criterion='roc_auc'):
        """
        Selecciona el mejor modelo basado en un criterio.
        
        Parameters:
        -----------
        criterion : str
            Métrica para seleccionar el mejor modelo
        """
        print(f"\n🏆 Seleccionando mejor modelo (criterio: {criterion})...")
        
        best_score = -1
        best_name = None
        
        for name, results in self.results.items():
            score = results['metrics'][criterion]
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]['trained_model']
        
        print(f"\n🥇 MEJOR MODELO: {best_name}")
        print(f"   Score ({criterion}): {best_score:.4f}")
        print("\n📊 Métricas del mejor modelo:")
        for metric, value in self.results[best_name]['metrics'].items():
            print(f"   {metric:<15}: {value:.4f}")
        
        return self.best_model, self.best_model_name
    
    
    def generate_report(self):
        """
        Genera un reporte completo de la evaluación.
        """
        print("\n📄 Generando reporte completo...")
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_model': self.best_model_name,
            'models_evaluated': list(self.results.keys()),
            'results_summary': {}
        }
        
        for name, results in self.results.items():
            report['results_summary'][name] = {
                'metrics': results['metrics'],
                'training_time': results['training_time']
            }
        
        # Guardar como JSON
        with open('../../outputs/evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        print("✅ Reporte guardado en outputs/evaluation_report.json")
        
        # Reporte de clasificación del mejor modelo
        print(f"\n📋 CLASSIFICATION REPORT - {self.best_model_name}:")
        print("=" * 80)
        print(self.results[self.best_model_name]['classification_report'])
        
        return report
    
    
    def save_best_model(self, output_dir='../../models'):
        """
        Guarda el mejor modelo.
        """
        print(f"\n💾 Guardando mejor modelo ({self.best_model_name})...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar modelo
        model_path = f'{output_dir}/best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Guardar metadata del modelo
        metadata = {
            'model_name': self.best_model_name,
            'model_type': str(type(self.best_model)),
            'metrics': self.results[self.best_model_name]['metrics'],
            'training_time': self.results[self.best_model_name]['training_time'],
            'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_used': list(self.X_train.columns)
        }
        
        metadata_path = f'{output_dir}/best_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✅ Modelo guardado en {model_path}")
        print(f"✅ Metadata guardado en {metadata_path}")
        
        return model_path, metadata_path


def summarize_classification(results_dict):
    """
    Función auxiliar para resumir resultados de clasificación.
    
    Parameters:
    -----------
    results_dict : dict
        Diccionario con resultados de modelos
    """
    print("\n" + "=" * 80)
    print("RESUMEN DE CLASIFICACIÓN")
    print("=" * 80)
    
    summary_data = []
    for model_name, results in results_dict.items():
        summary_data.append({
            'Modelo': model_name,
            'ROC-AUC': results['metrics']['roc_auc'],
            'PR-AUC': results['metrics']['pr_auc'],
            'F1-Score': results['metrics']['f1_score'],
            'Precision': results['metrics']['precision'],
            'Recall': results['metrics']['recall']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('ROC-AUC', ascending=False)
    
    display(df_summary)
    
    return df_summary


def build_model(X_train, y_train, model_type='xgboost'):
    """
    Función auxiliar para construir un modelo específico.
    
    Parameters:
    -----------
    X_train : DataFrame
        Features de entrenamiento
    y_train : Series
        Target de entrenamiento
    model_type : str
        Tipo de modelo a construir
    """
    if model_type.lower() == 'xgboost':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif model_type.lower() == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    elif model_type.lower() == 'lightgbm':
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    model.fit(X_train, y_train)
    
    return model


def main():
    """Función principal para ejecutar el pipeline completo."""
    
    # Crear instancia
    trainer = ModelTrainingEvaluation()
    
    # 1. Cargar datos
    trainer.load_data()
    
    # 2. Aplicar SMOTE para balanceo
    trainer.apply_smote(sampling_strategy=0.3)
    
    # 3. Definir modelos
    trainer.define_models()
    
    # 4. Entrenar modelos
    trainer.train_models()
    
    # 5. Evaluar modelos
    trainer.evaluate_models()
    
    # 6. Comparar modelos
    df_comparison = trainer.compare_models()
    
    # 7. Seleccionar mejor modelo
    trainer.select_best_model(criterion='roc_auc')
    
    # 8. Generar reporte
    report = trainer.generate_report()
    
    # 9. Guardar mejor modelo
    trainer.save_best_model()
    
    # Resumen final
    print("\n" + "=" * 80)
    print("MODEL TRAINING & EVALUATION COMPLETADO ✅")
    print("=" * 80)
    print(f"\n🥇 Mejor modelo: {trainer.best_model_name}")
    print(f"\n📁 Archivos generados:")
    print("  • models/best_model.pkl")
    print("  • models/best_model_metadata.json")
    print("  • outputs/model_comparison.csv")
    print("  • outputs/evaluation_report.json")
    print("  • outputs/metrics_comparison.png")
    print("  • outputs/roc_curves.png")
    print("  • outputs/pr_curves.png")
    print("  • outputs/confusion_matrices.png")
    print("\n➡️ Siguiente paso: Model Deployment")
    print("=" * 80)
    
    return trainer


if __name__ == "__main__":
    trainer = main()
