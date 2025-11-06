"""
Model Training and Evaluation Pipeline - Version Compatible con Windows
========================================================================
Entrena 5 modelos diferentes y selecciona el mejor
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
import json

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Métricas
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# Balanceo
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ENTRENAMIENTO Y EVALUACION DE MULTIPLES MODELOS")
print("=" * 80)

# 1. Cargar datos
print("\n[PASO 1] Cargando datos preprocesados...")
X_train = pd.read_pickle('../../data/processed/X_train.pkl')
X_test = pd.read_pickle('../../data/processed/X_test.pkl')
y_train = pd.read_pickle('../../data/processed/y_train.pkl')
y_test = pd.read_pickle('../../data/processed/y_test.pkl')
preprocessor = pd.read_pickle('../../data/processed/preprocessor.pkl')

print(f"[OK] Datos cargados - Train: {X_train.shape}, Test: {X_test.shape}")
print(f"     Fraude en train: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")

# 2. Transformar datos
print("\n[PASO 2] Transformando datos con preprocessor...")
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
print(f"[OK] Datos transformados - Shape: {X_train_transformed.shape}")

# 3. Aplicar SMOTE
print("\n[PASO 3] Aplicando SMOTE...")
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
print(f"[OK] SMOTE aplicado")
print(f"     Antes: No Fraude={len(y_train)-y_train.sum():,}, Fraude={y_train.sum():,}")
print(f"     Despues: No Fraude={len(y_train_resampled)-y_train_resampled.sum():,}, Fraude={y_train_resampled.sum():,}")

# 4. Definir modelos
print("\n[PASO 4] Definiendo modelos a entrenar...")
scale_pos_weight = (y_train_resampled==0).sum()/(y_train_resampled==1).sum()

models = {
    'Logistic_Regression': {
        'model': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'description': 'Modelo lineal simple y rapido'
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
        'description': 'Ensemble de arboles de decision'
    },
    
    'XGBoost': {
        'model': XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'description': 'Gradient Boosting optimizado (XGBoost)'
    },
    
    'LightGBM': {
        'model': LGBMClassifier(
            n_estimators=150,
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
        'description': 'Gradient Boosting ligero y rapido (LightGBM)'
    },
    
    'Gradient_Boosting': {
        'model': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'description': 'Gradient Boosting clasico (sklearn)'
    }
}

print(f"[OK] {len(models)} modelos definidos:")
for name, info in models.items():
    print(f"     - {name}: {info['description']}")

# 5. Entrenar y evaluar modelos
print("\n[PASO 5] Entrenando y evaluando modelos...")
print("=" * 80)

results = {}

for name, model_info in models.items():
    print(f"\n[ENTRENANDO] {name}...")
    start_time = datetime.now()
    
    # Entrenar
    model = model_info['model']
    model.fit(X_train_resampled, y_train_resampled)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Predecir
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    # Guardar resultados
    results[name] = {
        'model': model,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'training_time': training_time,
        'description': model_info['description']
    }
    
    print(f"[OK] {name} - Entrenado en {training_time:.2f}s")
    print(f"     ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

# 6. Comparar modelos
print("\n" + "=" * 80)
print("[PASO 6] Comparacion de modelos:")
print("=" * 80)

comparison_data = []
for name, res in results.items():
    row = {'Model': name}
    row.update(res['metrics'])
    row['Training_Time'] = res['training_time']
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('roc_auc', ascending=False)

print("\nTabla de comparacion (ordenado por ROC-AUC):")
print("-" * 120)
print(df_comparison.to_string(index=False))
print("-" * 120)

# 7. Seleccionar mejor modelo
print("\n[PASO 7] Seleccionando mejor modelo...")
best_model_name = df_comparison.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_metrics = results[best_model_name]['metrics']

print(f"\n[MEJOR MODELO] {best_model_name}")
print(f"  ROC-AUC:     {best_metrics['roc_auc']:.4f}")
print(f"  PR-AUC:      {best_metrics['pr_auc']:.4f}")
print(f"  F1-Score:    {best_metrics['f1_score']:.4f}")
print(f"  Precision:   {best_metrics['precision']:.4f}")
print(f"  Recall:      {best_metrics['recall']:.4f}")
print(f"  Accuracy:    {best_metrics['accuracy']:.4f}")
print(f"  Tiempo:      {results[best_model_name]['training_time']:.2f}s")

# 8. Guardar resultados
print("\n[PASO 8] Guardando resultados...")
os.makedirs('../../models', exist_ok=True)
os.makedirs('../../outputs', exist_ok=True)

# Guardar mejor modelo
with open('../../models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("[OK] Mejor modelo guardado: models/best_model.pkl")

# Guardar metadata del mejor modelo
best_metadata = {
    'model_name': best_model_name,
    'metrics': best_metrics,
    'training_time': results[best_model_name]['training_time'],
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'description': results[best_model_name]['description']
}

with open('../../models/best_model_metadata.json', 'w') as f:
    json.dump(best_metadata, f, indent=4)
print("[OK] Metadata guardada: models/best_model_metadata.json")

# Guardar comparación de modelos
df_comparison.to_csv('../../outputs/model_comparison.csv', index=False)
print("[OK] Comparacion guardada: outputs/model_comparison.csv")

# Guardar todos los resultados (sin los modelos para ahorrar espacio)
results_to_save = {}
for name, res in results.items():
    results_to_save[name] = {
        'metrics': res['metrics'],
        'confusion_matrix': res['confusion_matrix'],
        'training_time': res['training_time'],
        'description': res['description']
    }

with open('../../outputs/all_models_results.json', 'w') as f:
    json.dump(results_to_save, f, indent=4)
print("[OK] Resultados completos guardados: outputs/all_models_results.json")

# Guardar también el modelo XGBoost específicamente (para compatibilidad)
if 'XGBoost' in results:
    with open('../../models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(results['XGBoost']['model'], f)
    print("[OK] Modelo XGBoost guardado: models/xgboost_model.pkl")

# 9. Generar predicciones para monitoreo
print("\n[PASO 9] Generando predicciones para monitoreo...")
predictions_df = pd.DataFrame({
    'true_label': y_test.values if hasattr(y_test, 'values') else y_test,
    'predicted_label': results[best_model_name]['y_pred'],
    'predicted_proba': results[best_model_name]['y_pred_proba']
})

os.makedirs('../../outputs/monitoring', exist_ok=True)
predictions_df.to_csv('../../outputs/monitoring/predictions.csv', index=False)
print("[OK] Predicciones guardadas: outputs/monitoring/predictions.csv")

# Resumen final
print("\n" + "=" * 80)
print("[COMPLETADO] ENTRENAMIENTO DE MODELOS EXITOSO")
print("=" * 80)
print(f"\nMejor modelo: {best_model_name}")
print(f"ROC-AUC Score: {best_metrics['roc_auc']:.4f}")
print("\nArchivos generados:")
print("  - models/best_model.pkl")
print("  - models/best_model_metadata.json")
print("  - models/xgboost_model.pkl")
print("  - outputs/model_comparison.csv")
print("  - outputs/all_models_results.json")
print("  - outputs/monitoring/predictions.csv")
print("\nSiguiente paso: python model_monitoring.py")
print("=" * 80)
