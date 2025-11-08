"""
API de Predicción de Fraude - FastAPI
======================================

API RESTful para servir el modelo de detección de fraude entrenado.
Soporta predicciones individuales y por lotes.

Endpoints:
----------
- GET /: Información de la API
- GET /health: Estado de la API
- GET /model/info: Información del modelo cargado
- POST /predict: Predicción individual
- POST /predict/batch: Predicción por lotes (múltiples registros)

Autor: MLOps Team
Universidad Católica Luis Amigó
Fecha: 2025-11-07
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import io
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================

app = FastAPI(
    title="Fraud Detection API",
    description="API para detección de fraude en transacciones financieras utilizando Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# MODELOS DE DATOS (PYDANTIC)
# ============================================================================

class Transaction(BaseModel):
    """
    Modelo de datos para una transacción individual.
    
    Attributes:
        step: Paso temporal de la transacción (1 hora)
        type: Tipo de transacción (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
        amount: Monto de la transacción
        nameOrig: ID del cliente que origina la transacción
        oldbalanceOrg: Balance inicial del origen
        newbalanceOrig: Nuevo balance del origen
        nameDest: ID del cliente destino
        oldbalanceDest: Balance inicial del destino
        newbalanceDest: Nuevo balance del destino
    """
    step: int = Field(..., ge=1, description="Paso temporal (≥1)")
    type: str = Field(..., description="Tipo de transacción")
    amount: float = Field(..., ge=0, description="Monto de transacción (≥0)")
    nameOrig: str = Field(..., description="ID cliente origen")
    oldbalanceOrg: float = Field(..., ge=0, description="Balance inicial origen")
    newbalanceOrig: float = Field(..., ge=0, description="Nuevo balance origen")
    nameDest: str = Field(..., description="ID cliente destino")
    oldbalanceDest: float = Field(..., ge=0, description="Balance inicial destino")
    newbalanceDest: float = Field(..., ge=0, description="Nuevo balance destino")
    
    @validator('type')
    def validate_transaction_type(cls, v):
        """Valida que el tipo de transacción sea válido"""
        valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        if v not in valid_types:
            raise ValueError(f'Tipo de transacción debe ser uno de: {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class TransactionBatch(BaseModel):
    """
    Modelo para lote de transacciones.
    
    Attributes:
        transactions: Lista de transacciones a predecir
    """
    transactions: List[Transaction] = Field(..., min_items=1, description="Lista de transacciones")
    
    class Config:
        schema_extra = {
            "example": {
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
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Respuesta de predicción individual.
    
    Attributes:
        is_fraud: 1 si es fraude, 0 si no
        fraud_probability: Probabilidad de ser fraude (0-1)
        risk_level: Nivel de riesgo (LOW, MEDIUM, HIGH, CRITICAL)
        timestamp: Timestamp de la predicción
        model_version: Versión del modelo usado
    """
    is_fraud: int = Field(..., description="1=Fraude, 0=Legítimo")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probabilidad de fraude")
    risk_level: str = Field(..., description="Nivel de riesgo")
    timestamp: str = Field(..., description="Timestamp de predicción")
    model_version: str = Field(..., description="Versión del modelo")
    
    class Config:
        schema_extra = {
            "example": {
                "is_fraud": 0,
                "fraud_probability": 0.023,
                "risk_level": "LOW",
                "timestamp": "2025-11-07T14:30:15.123456",
                "model_version": "Random_Forest_v1.0"
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Respuesta de predicción por lotes.
    
    Attributes:
        predictions: Lista de predicciones
        total_transactions: Total de transacciones procesadas
        frauds_detected: Número de fraudes detectados
        fraud_rate: Tasa de fraude (%)
        processing_time_ms: Tiempo de procesamiento en ms
        timestamp: Timestamp de la predicción
        model_version: Versión del modelo usado
    """
    predictions: List[PredictionResponse] = Field(..., description="Lista de predicciones")
    total_transactions: int = Field(..., description="Total de transacciones")
    frauds_detected: int = Field(..., description="Fraudes detectados")
    fraud_rate: float = Field(..., description="Tasa de fraude %")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento (ms)")
    timestamp: str = Field(..., description="Timestamp de predicción")
    model_version: str = Field(..., description="Versión del modelo")


# ============================================================================
# CARGA DEL MODELO Y PREPROCESADOR
# ============================================================================

class ModelLoader:
    """Clase para cargar y mantener en memoria el modelo y preprocesador"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_metadata = None
        self.model_loaded = False
        
    def load_model(self, model_path: str = '../models/best_model.pkl'):
        """
        Carga el modelo entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo
        """
        try:
            model_file = Path(model_path)
            
            if not model_file.exists():
                logger.error(f"Modelo no encontrado en: {model_file.absolute()}")
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"✅ Modelo cargado desde: {model_file.absolute()}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def load_preprocessor(self, preprocessor_path: str = '../data/processed/preprocessor.pkl'):
        """
        Carga el preprocesador.
        
        Args:
            preprocessor_path: Ruta al preprocesador
        """
        try:
            prep_file = Path(preprocessor_path)
            
            if not prep_file.exists():
                logger.error(f"Preprocesador no encontrado en: {prep_file.absolute()}")
                raise FileNotFoundError(f"Preprocesador no encontrado: {preprocessor_path}")
            
            with open(prep_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            logger.info(f"✅ Preprocesador cargado desde: {prep_file.absolute()}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando preprocesador: {e}")
            raise
    
    def load_metadata(self, metadata_path: str = '../models/best_model_metadata.json'):
        """
        Carga la metadata del modelo.
        
        Args:
            metadata_path: Ruta al archivo de metadata
        """
        try:
            meta_file = Path(metadata_path)
            
            if not meta_file.exists():
                logger.warning(f"Metadata no encontrada: {meta_file.absolute()}")
                self.model_metadata = {
                    'model_name': 'Unknown',
                    'model_type': 'Unknown',
                    'metrics': {},
                    'trained_on': 'Unknown'
                }
                return False
            
            with open(meta_file, 'r') as f:
                self.model_metadata = json.load(f)
            
            logger.info(f"✅ Metadata cargada desde: {meta_file.absolute()}")
            return True
            
        except Exception as e:
            logger.warning(f"Error cargando metadata: {e}")
            self.model_metadata = {}
            return False
    
    def initialize(self):
        """Inicializa el modelo, preprocesador y metadata"""
        try:
            logger.info("🔄 Iniciando carga de modelo...")
            
            self.load_model()
            self.load_preprocessor()
            self.load_metadata()
            
            self.model_loaded = True
            logger.info("✅ Modelo inicializado correctamente")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando modelo: {e}")
            self.model_loaded = False
            return False


# Instancia global del loader
model_loader = ModelLoader()


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_risk_level(probability: float) -> str:
    """
    Calcula el nivel de riesgo basado en la probabilidad.
    
    Args:
        probability: Probabilidad de fraude (0-1)
    
    Returns:
        Nivel de riesgo: LOW, MEDIUM, HIGH, CRITICAL
    """
    if probability < 0.3:
        return "LOW"
    elif probability < 0.5:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


def preprocess_transaction(transaction: Transaction) -> pd.DataFrame:
    """
    Preprocesa una transacción para predicción.
    
    Args:
        transaction: Objeto Transaction
    
    Returns:
        DataFrame preprocesado
    """
    # Convertir a diccionario
    data = transaction.dict()
    
    # Crear DataFrame
    df = pd.DataFrame([data])
    
    # Aplicar preprocesador (feature engineering)
    if model_loader.preprocessor is not None:
        df_processed = model_loader.preprocessor.transform(df)
        return df_processed
    else:
        # Si no hay preprocesador, usar datos raw (no recomendado)
        logger.warning("⚠️ Preprocesador no disponible, usando datos raw")
        return df


def predict_single(transaction: Transaction) -> PredictionResponse:
    """
    Realiza predicción para una transacción individual.
    
    Args:
        transaction: Transacción a predecir
    
    Returns:
        Respuesta de predicción
    """
    # Preprocesar
    X = preprocess_transaction(transaction)
    
    # Predecir
    prediction = int(model_loader.model.predict(X)[0])
    probability = float(model_loader.model.predict_proba(X)[0, 1])
    
    # Calcular nivel de riesgo
    risk_level = calculate_risk_level(probability)
    
    # Crear respuesta
    response = PredictionResponse(
        is_fraud=prediction,
        fraud_probability=round(probability, 4),
        risk_level=risk_level,
        timestamp=datetime.now().isoformat(),
        model_version=model_loader.model_metadata.get('model_name', 'Unknown')
    )
    
    return response


def predict_batch(transactions: List[Transaction]) -> BatchPredictionResponse:
    """
    Realiza predicción para múltiples transacciones.
    
    Args:
        transactions: Lista de transacciones
    
    Returns:
        Respuesta de predicción por lotes
    """
    start_time = datetime.now()
    
    # Convertir a DataFrame
    data = [t.dict() for t in transactions]
    df = pd.DataFrame(data)
    
    # Preprocesar
    if model_loader.preprocessor is not None:
        X = model_loader.preprocessor.transform(df)
    else:
        X = df
    
    # Predecir
    predictions = model_loader.model.predict(X).astype(int)
    probabilities = model_loader.model.predict_proba(X)[:, 1]
    
    # Crear respuestas individuales
    prediction_list = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        response = PredictionResponse(
            is_fraud=int(pred),
            fraud_probability=round(float(prob), 4),
            risk_level=calculate_risk_level(prob),
            timestamp=datetime.now().isoformat(),
            model_version=model_loader.model_metadata.get('model_name', 'Unknown')
        )
        prediction_list.append(response)
    
    # Calcular estadísticas
    total = len(predictions)
    frauds = int(predictions.sum())
    fraud_rate = (frauds / total * 100) if total > 0 else 0.0
    
    # Tiempo de procesamiento
    processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
    
    # Crear respuesta batch
    batch_response = BatchPredictionResponse(
        predictions=prediction_list,
        total_transactions=total,
        frauds_detected=frauds,
        fraud_rate=round(fraud_rate, 2),
        processing_time_ms=round(processing_time, 2),
        timestamp=datetime.now().isoformat(),
        model_version=model_loader.model_metadata.get('model_name', 'Unknown')
    )
    
    return batch_response


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento ejecutado al iniciar la API"""
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO API DE DETECCIÓN DE FRAUDE")
    logger.info("=" * 80)
    
    # Cargar modelo
    success = model_loader.initialize()
    
    if success:
        logger.info("✅ API lista para recibir solicitudes")
    else:
        logger.error("❌ Error inicializando API")
    
    logger.info("=" * 80)


@app.get("/", tags=["Info"])
async def root():
    """
    Endpoint raíz - Información general de la API.
    
    Returns:
        Información de la API
    """
    return {
        "api": "Fraud Detection API",
        "version": "1.0.0",
        "description": "API para detección de fraude en transacciones financieras",
        "university": "Universidad Católica Luis Amigó",
        "author": "DANIEL ALEJANDRO RINCON VALENCIA",
        "endpoints": {
            "/": "Información de la API",
            "/health": "Estado de salud",
            "/model/info": "Información del modelo",
            "/predict": "Predicción individual (POST)",
            "/predict/batch": "Predicción por lotes (POST)",
            "/predict/csv": "Predicción desde CSV (POST)"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """
    Endpoint de salud - Verifica que la API esté funcionando.
    
    Returns:
        Estado de salud de la API
    """
    return {
        "status": "healthy" if model_loader.model_loaded else "unhealthy",
        "model_loaded": model_loader.model_loaded,
        "timestamp": datetime.now().isoformat(),
        "uptime": "running"
    }


@app.get("/model/info", tags=["Info"])
async def model_info():
    """
    Información del modelo cargado.
    
    Returns:
        Metadata del modelo
    """
    if not model_loader.model_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "model_name": model_loader.model_metadata.get('model_name', 'Unknown'),
        "model_type": model_loader.model_metadata.get('model_type', 'Unknown'),
        "metrics": model_loader.model_metadata.get('metrics', {}),
        "trained_on": model_loader.model_metadata.get('trained_on', 'Unknown'),
        "features_count": len(model_loader.model_metadata.get('features_used', [])),
        "status": "loaded"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(transaction: Transaction):
    """
    Predicción individual - Predice si una transacción es fraudulenta.
    
    Args:
        transaction: Datos de la transacción
    
    Returns:
        Predicción con probabilidad y nivel de riesgo
    
    Raises:
        HTTPException: Si el modelo no está cargado o hay error en predicción
    """
    if not model_loader.model_loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        logger.info(f"📊 Predicción individual: {transaction.nameOrig} → {transaction.nameDest}")
        
        prediction = predict_single(transaction)
        
        logger.info(f"✅ Resultado: {'FRAUDE' if prediction.is_fraud else 'LEGÍTIMO'} "
                   f"(prob={prediction.fraud_probability:.4f})")
        
        return prediction
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch_endpoint(batch: TransactionBatch):
    """
    Predicción por lotes - Predice múltiples transacciones en una sola solicitud.
    
    Args:
        batch: Lote de transacciones
    
    Returns:
        Predicciones y estadísticas del lote
    
    Raises:
        HTTPException: Si el modelo no está cargado o hay error en predicción
    """
    if not model_loader.model_loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        logger.info(f"📊 Predicción batch: {len(batch.transactions)} transacciones")
        
        batch_result = predict_batch(batch.transactions)
        
        logger.info(f"✅ Procesadas {batch_result.total_transactions} transacciones en "
                   f"{batch_result.processing_time_ms:.2f}ms")
        logger.info(f"   Fraudes detectados: {batch_result.frauds_detected} "
                   f"({batch_result.fraud_rate:.2f}%)")
        
        return batch_result
        
    except Exception as e:
        logger.error(f"❌ Error en predicción batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")


@app.post("/predict/csv", tags=["Predictions"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predicción desde CSV - Carga un archivo CSV y predice todas las transacciones.
    
    Args:
        file: Archivo CSV con transacciones
    
    Returns:
        Predicciones en formato JSON
    
    Raises:
        HTTPException: Si hay error leyendo CSV o en predicción
    """
    if not model_loader.model_loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Validar extensión
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser CSV")
    
    try:
        # Leer CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        logger.info(f"📊 Predicción desde CSV: {len(df)} transacciones")
        
        # Validar columnas requeridas
        required_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Columnas faltantes en CSV: {missing_cols}"
            )
        
        # Convertir a Transaction objects
        transactions = []
        for _, row in df.iterrows():
            try:
                t = Transaction(**row.to_dict())
                transactions.append(t)
            except Exception as e:
                logger.warning(f"⚠️ Fila inválida ignorada: {e}")
        
        # Predecir
        batch_result = predict_batch(transactions)
        
        logger.info(f"✅ Procesadas {batch_result.total_transactions} transacciones desde CSV")
        
        return batch_result
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")
    except Exception as e:
        logger.error(f"❌ Error procesando CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando CSV: {str(e)}")


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuración del servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )
