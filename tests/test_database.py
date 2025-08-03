"""
Unit tests for database functionality
"""
import pytest
import sys
import os
import tempfile
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import PredictionLogger

@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    logger = PredictionLogger(db_path)
    yield logger
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass

def test_database_initialization(temp_db):
    """Test database initialization"""
    logger = temp_db
    
    # Database should be initialized without errors
    assert os.path.exists(logger.db_path)

def test_log_prediction(temp_db):
    """Test logging predictions"""
    logger = temp_db
    
    # Log a test prediction
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    logger.log_prediction(
        input_data=input_data,
        prediction="setosa",
        probability=0.95,
        model_version="1.0.0",
        processing_time_ms=10.5,
        request_id="test-123",
        client_ip="127.0.0.1",
        user_agent="test-agent"
    )
    
    # Retrieve predictions
    predictions = logger.get_predictions(limit=1)
    assert len(predictions) == 1
    
    pred = predictions[0]
    assert pred['prediction'] == "setosa"
    assert pred['probability'] == 0.95
    assert pred['model_version'] == "1.0.0"
    assert pred['processing_time_ms'] == 10.5
    assert pred['request_id'] == "test-123"
    assert pred['input_data'] == input_data

def test_log_model_metric(temp_db):
    """Test logging model metrics"""
    logger = temp_db
    
    # Log a test metric
    metadata = {
        "training_samples": 120,
        "test_samples": 30,
        "features": 4
    }
    
    logger.log_model_metric(
        model_name="SVM",
        metric_name="accuracy",
        metric_value=0.96,
        metadata=metadata
    )
    
    # Retrieve metrics
    metrics = logger.get_training_metrics("SVM")
    assert len(metrics) == 1
    
    metric = metrics[0]
    assert metric['model_name'] == "SVM"
    assert metric['metric_name'] == "accuracy"
    assert metric['metric_value'] == 0.96
    assert metric['metadata'] == metadata

def test_log_api_metric(temp_db):
    """Test logging API metrics"""
    logger = temp_db
    
    logger.log_api_metric(
        endpoint="/predict",
        method="POST",
        status_code=200,
        response_time_ms=15.5,
        client_ip="127.0.0.1"
    )
    
    # This test just ensures no errors occur
    # We don't have a getter for API metrics in the current implementation

def test_get_prediction_stats(temp_db):
    """Test getting prediction statistics"""
    logger = temp_db
    
    # Log some test predictions
    test_predictions = [
        ("setosa", 0.95),
        ("setosa", 0.98),
        ("versicolor", 0.85),
        ("virginica", 0.92)
    ]
    
    for pred, prob in test_predictions:
        logger.log_prediction(
            input_data={"sepal_length": 5.0, "sepal_width": 3.0, "petal_length": 1.0, "petal_width": 0.5},
            prediction=pred,
            probability=prob,
            model_version="1.0.0",
            processing_time_ms=10.0
        )
    
    stats = logger.get_prediction_stats()
    
    assert stats['total_predictions'] == 4
    assert stats['predictions_by_class']['setosa'] == 2
    assert stats['predictions_by_class']['versicolor'] == 1
    assert stats['predictions_by_class']['virginica'] == 1
    assert stats['avg_processing_time_ms'] == 10.0

def test_get_best_model_info(temp_db):
    """Test getting best model information"""
    logger = temp_db
    
    # Log a model selection metric
    metadata = {
        "selection_criteria": "f1_score",
        "all_model_scores": {
            "SVM": 0.96,
            "Random Forest": 0.89,
            "Logistic Regression": 0.93
        }
    }
    
    logger.log_model_metric(
        model_name="BEST_MODEL_SVM",
        metric_name="model_selection",
        metric_value=0.96,
        metadata=metadata
    )
    
    best_model_info = logger.get_best_model_info()
    
    assert best_model_info['model_name'] == "BEST_MODEL_SVM"
    assert best_model_info['metric_name'] == "model_selection"
    assert best_model_info['metric_value'] == 0.96
    assert best_model_info['metadata'] == metadata

def test_get_training_metrics_all(temp_db):
    """Test getting all training metrics"""
    logger = temp_db
    
    # Log metrics for multiple models
    models_metrics = [
        ("SVM", "accuracy", 0.96),
        ("SVM", "f1_score", 0.95),
        ("Random Forest", "accuracy", 0.89),
        ("Random Forest", "f1_score", 0.88)
    ]
    
    for model, metric, value in models_metrics:
        logger.log_model_metric(model, metric, value)
    
    # Get all metrics
    all_metrics = logger.get_training_metrics()
    assert len(all_metrics) == 4
    
    # Get metrics for specific model
    svm_metrics = logger.get_training_metrics("SVM")
    assert len(svm_metrics) == 2
    
    metric_names = [m['metric_name'] for m in svm_metrics]
    assert "accuracy" in metric_names
    assert "f1_score" in metric_names

def test_empty_database_queries(temp_db):
    """Test queries on empty database"""
    logger = temp_db
    
    # Test empty queries
    predictions = logger.get_predictions()
    assert predictions == []
    
    stats = logger.get_prediction_stats()
    assert stats['total_predictions'] == 0
    
    metrics = logger.get_training_metrics()
    assert metrics == []
    
    best_model = logger.get_best_model_info()
    assert best_model == {}

def test_json_serialization(temp_db):
    """Test JSON serialization/deserialization"""
    logger = temp_db
    
    # Test complex input data
    complex_input = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "extra_info": {
            "source": "test",
            "timestamp": "2025-08-03T12:00:00"
        }
    }
    
    logger.log_prediction(
        input_data=complex_input,
        prediction="setosa",
        probability=0.95,
        model_version="1.0.0",
        processing_time_ms=10.0
    )
    
    predictions = logger.get_predictions(limit=1)
    assert len(predictions) == 1
    
    # Check that complex data is properly serialized/deserialized
    retrieved_input = predictions[0]['input_data']
    assert retrieved_input == complex_input
    assert retrieved_input['extra_info']['source'] == "test"
