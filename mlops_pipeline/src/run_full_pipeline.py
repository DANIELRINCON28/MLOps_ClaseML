"""
Script completo para ejecutar todo el pipeline MLOps
=====================================================
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PIPELINE MLOPS COMPLETO - DETECCIÓN DE FRAUDE")
print("=" * 80)

# 1. Cargar datos
print("\n📂 PASO 1: Cargando datos...")
with open('../../data/processed/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('../../data/processed/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('../../data/processed/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('../../data/processed/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('../../data/processed/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

print(f"✅ Datos cargados - Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   Fraude en train: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")

# 2. Transformar datos
print("\n🔧 PASO 2: Transformando datos con preprocessor...")
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
print(f"✅ Datos transformados - Shape: {X_train_transformed.shape}")

# 3. Aplicar SMOTE
print("\n⚖️ PASO 3: Aplicando SMOTE...")
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
print(f"✅ SMOTE aplicado")
print(f"   Antes: No Fraude={len(y_train)-y_train.sum():,}, Fraude={y_train.sum():,}")
print(f"   Después: No Fraude={len(y_train_resampled)-y_train_resampled.sum():,}, Fraude={y_train_resampled.sum():,}")

# 4. Entrenar modelo XGBoost
print("\n🤖 PASO 4: Entrenando modelo XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

model.fit(X_train_resampled, y_train_resampled)
print("✅ Modelo entrenado")

# 5. Evaluar modelo
print("\n📊 PASO 5: Evaluando modelo...")
y_pred = model.predict(X_test_transformed)
y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"✅ ROC-AUC Score: {roc_auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 6. Guardar modelo
print("\n💾 PASO 6: Guardando modelo y resultados...")
import os
os.makedirs('../../models', exist_ok=True)

with open('../../models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Guardar métricas
metrics = {
    'model_name': 'XGBoost',
    'roc_auc': float(roc_auc),
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1
}

with open('../../models/model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("✅ Modelo guardado en ../../models/xgboost_model.pkl")

# 7. Generar predicciones para monitoreo
print("\n🔮 PASO 7: Generando predicciones para monitoreo...")
predictions_df = pd.DataFrame({
    'true_label': y_test.values if hasattr(y_test, 'values') else y_test,
    'predicted_label': y_pred,
    'predicted_proba': y_pred_proba
})

os.makedirs('../../outputs/monitoring', exist_ok=True)
predictions_df.to_csv('../../outputs/monitoring/predictions.csv', index=False)
print("✅ Predicciones guardadas")

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETADO CON ÉXITO")
print("=" * 80)
print(f"\n📊 Resultados finales:")
print(f"   • ROC-AUC: {roc_auc:.4f}")
print(f"   • Modelo guardado: models/xgboost_model.pkl")
print(f"   • Predicciones: outputs/monitoring/predictions.csv")
print(f"\n➡️  Siguiente paso: python model_monitoring.py")
print("=" * 80)
