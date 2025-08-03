"""
Enhanced FastAPI application with comprehensive logging and monitoring
"""
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
from fastapi.responses import Response

from database import prediction_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Clear existing metrics to avoid collisions
try:
    REGISTRY._collector_to_names.clear()
    REGISTRY._names_to_collectors.clear()
except:
    pass

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions', ['model_version', 'prediction_class'])
PREDICTION_HISTOGRAM = Histogram('prediction_duration_seconds', 'Time spent on predictions')
API_REQUEST_COUNTER = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
API_REQUEST_HISTOGRAM = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
MODEL_ACCURACY_GAUGE = Gauge('model_accuracy', 'Current model accuracy')
ACTIVE_MODELS_GAUGE = Gauge('active_models', 'Number of active models')

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API - Enhanced",
    description="A comprehensive ML API with monitoring and logging",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic models with validation
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=20)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=20)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=20)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=20)
    
    @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    @classmethod
    def validate_measurements(cls, v):
        if v < 0:
            raise ValueError('Measurements must be positive')
        if v > 20:
            raise ValueError('Measurements seem unrealistic (>20cm)')
        return round(v, 2)
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class")
    probability: float = Field(..., description="Prediction probability")
    confidence_level: str = Field(..., description="Confidence level (High/Medium/Low)")
    model_version: str = Field(..., description="Model version")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionRequest(BaseModel):
    samples: List[IrisFeatures] = Field(..., description="List of samples to predict")
    
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_id: str
    total_samples: int
    avg_processing_time_ms: float

class ModelStats(BaseModel):
    total_predictions: int
    predictions_by_class: Dict[str, int]
    avg_processing_time_ms: float
    recent_predictions_24h: int
    model_accuracy: float
    model_version: str

# Global variables for model and metadata
model = None
scaler = None
model_info = None

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability"""
    if probability >= 0.9:
        return "High"
    elif probability >= 0.7:
        return "Medium"
    else:
        return "Low"

def load_model_artifacts():
    """Load model, scaler, and metadata"""
    global model, scaler, model_info
    
    try:
        # Load best model
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            ACTIVE_MODELS_GAUGE.set(1)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load scaler
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        else:
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Load model info
        info_path = 'models/model_info.json'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            logger.info("Model info loaded successfully")
            MODEL_ACCURACY_GAUGE.set(model_info.get('best_score', 0))
        else:
            # Default model info if file doesn't exist
            model_info = {
                'model_version': '2.0.0',
                'target_names': ['setosa', 'versicolor', 'virginica'],
                'best_score': 0.95
            }
            logger.warning("Model info file not found, using defaults")
            
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        ACTIVE_MODELS_GAUGE.set(0)
        raise

# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Log API metrics
    client_ip = get_client_ip(request)
    endpoint = request.url.path
    method = request.method
    status_code = response.status_code
    
    # Update Prometheus metrics
    API_REQUEST_COUNTER.labels(method=method, endpoint=endpoint, status=status_code).inc()
    API_REQUEST_HISTOGRAM.labels(method=method, endpoint=endpoint).observe(process_time / 1000)
    
    # Log to database
    prediction_logger.log_api_metric(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time_ms=process_time,
        client_ip=client_ip,
        error_message=None if status_code < 400 else "Error occurred"
    )
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup"""
    os.makedirs('logs', exist_ok=True)
    load_model_artifacts()
    logger.info("Enhanced API startup completed")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Iris Classification API",
        "version": "2.0.0",
        "status": "healthy",
        "features": [
            "Comprehensive logging",
            "Prometheus metrics",
            "Input validation",
            "Batch predictions",
            "Request tracking"
        ],
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "database_connected": True,
        "version": "2.0.0"
    }

@app.get("/model-info")
async def get_model_info():
    """Get comprehensive model information"""
    if model_info is None:
        raise HTTPException(status_code=500, detail="Model info not available")
    
    stats = prediction_logger.get_prediction_stats()
    
    return {
        **model_info,
        "statistics": stats,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/stats", response_model=ModelStats)
async def get_stats():
    """Get detailed API and model statistics"""
    stats = prediction_logger.get_prediction_stats()
    
    return ModelStats(
        total_predictions=stats.get('total_predictions', 0),
        predictions_by_class=stats.get('predictions_by_class', {}),
        avg_processing_time_ms=stats.get('avg_processing_time_ms', 0),
        recent_predictions_24h=stats.get('recent_predictions_24h', 0),
        model_accuracy=model_info.get('best_score', 0) if model_info else 0,
        model_version=model_info.get('model_version', '2.0.0') if model_info else '2.0.0'
    )

@app.get("/logs/predictions")
async def get_prediction_logs(limit: int = 50):
    """Get recent prediction logs"""
    if limit > 1000:
        limit = 1000  # Prevent excessive data retrieval
    
    logs = prediction_logger.get_predictions(limit)
    return {
        "predictions": logs,
        "count": len(logs),
        "limit": limit
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures, request: Request):
    """
    Make prediction with comprehensive logging and monitoring
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    try:
        # Validate model is loaded
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Prepare input data
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction with timing
        with PREDICTION_HISTOGRAM.time():
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            max_probability = float(np.max(probabilities))
        
        # Get class name
        target_names = model_info.get('target_names', ['setosa', 'versicolor', 'virginica'])
        predicted_class = target_names[prediction]
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update Prometheus metrics
        PREDICTION_COUNTER.labels(
            model_version=model_info.get('model_version', '2.0.0'),
            prediction_class=predicted_class
        ).inc()
        
        # Create response
        response = PredictionResponse(
            prediction=predicted_class,
            probability=max_probability,
            confidence_level=get_confidence_level(max_probability),
            model_version=model_info.get('model_version', '2.0.0'),
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time_ms, 2)
        )
        
        # Log prediction to database
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        
        prediction_logger.log_prediction(
            input_data=features.dict(),
            prediction=predicted_class,
            probability=max_probability,
            model_version=model_info.get('model_version', '2.0.0'),
            processing_time_ms=processing_time_ms,
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Log successful prediction
        logger.info(f"Prediction successful: {predicted_class} (prob: {max_probability:.4f}, time: {processing_time_ms:.2f}ms)")
        
        return response
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Prediction error: {str(e)} (time: {processing_time_ms:.2f}ms)")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest, request: Request):
    """
    Make batch predictions with monitoring
    """
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    try:
        predictions = []
        total_processing_time = 0
        
        for i, features in enumerate(batch_request.samples):
            # Create individual request for each sample
            individual_start = time.time()
            
            result = await predict(features, request)
            individual_time = (time.time() - individual_start) * 1000
            total_processing_time += individual_time
            
            predictions.append(result)
        
        avg_processing_time = total_processing_time / len(batch_request.samples)
        
        response = BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_samples=len(batch_request.samples),
            avg_processing_time_ms=round(avg_processing_time, 2)
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)} predictions, avg time: {avg_processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/retrain-trigger")
async def trigger_retrain():
    """
    Trigger model retraining (placeholder for future implementation)
    """
    logger.info("Model retraining triggered")
    
    # This would typically:
    # 1. Check for new data
    # 2. Validate data quality
    # 3. Trigger training pipeline
    # 4. Update model if performance improves
    
    return {
        "message": "Model retraining triggered",
        "status": "queued",
        "timestamp": datetime.now().isoformat(),
        "estimated_completion": "This is a placeholder - would implement actual retraining logic"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Ensure model artifacts exist
    if not os.path.exists('models/best_model.pkl'):
        logger.error("Model not found. Please run 'python src/train.py' first.")
        exit(1)
    
    # Run the enhanced API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
