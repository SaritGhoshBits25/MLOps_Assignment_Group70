"""
Unit tests for model training functionality
"""
import pytest
import sys
import os
import tempfile
import json
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from database import PredictionLogger

def train_model_simple(model, model_name, X_train, X_test, y_train, y_test, target_names, prediction_logger):
    """
    Simplified train_model function for testing without MLflow
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics to database
    training_metadata = {
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X_train[0]) if hasattr(X_train, '__len__') else X_train.shape[1],
        'model_params': model.get_params(),
        'target_classes': list(target_names)
    }
    
    prediction_logger.log_model_metric(model_name, 'accuracy', accuracy, training_metadata)
    prediction_logger.log_model_metric(model_name, 'precision', precision, training_metadata)
    prediction_logger.log_model_metric(model_name, 'recall', recall, training_metadata)
    prediction_logger.log_model_metric(model_name, 'f1_score', f1, training_metadata)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return model, metrics

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create a simple dataset
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_classes=3,
        n_informative=4,
        n_redundant=0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    target_names = ['class_0', 'class_1', 'class_2']
    
    return X_train, X_test, y_train, y_test, feature_names, target_names

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

def test_train_model_basic(sample_data, temp_db):
    """Test basic model training functionality"""
    X_train, X_test, y_train, y_test, feature_names, target_names = sample_data
    
    # Create a simple model
    model = LogisticRegression(random_state=42, max_iter=200)
    
    # Train the model
    trained_model, metrics = train_model_simple(
        model=model,
        model_name="Test_Logistic_Regression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        prediction_logger=temp_db
    )
    
    # Check that model is trained
    assert hasattr(trained_model, 'coef_')
    assert hasattr(trained_model, 'intercept_')
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    # Check metric values are reasonable
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1

def test_train_model_database_logging(sample_data, temp_db):
    """Test that training metrics are logged to database"""
    X_train, X_test, y_train, y_test, feature_names, target_names = sample_data
    
    model = LogisticRegression(random_state=42, max_iter=200)
    model_name = "Test_Model_DB_Logging"
    
    # Train the model
    trained_model, metrics = train_model_simple(
        model=model,
        model_name=model_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        prediction_logger=temp_db
    )
    
    # Check that metrics were logged to database
    db_metrics = temp_db.get_training_metrics(model_name)
    
    # Should have 4 metrics logged (accuracy, precision, recall, f1_score)
    assert len(db_metrics) == 4
    
    metric_names = [m['metric_name'] for m in db_metrics]
    assert 'accuracy' in metric_names
    assert 'precision' in metric_names
    assert 'recall' in metric_names
    assert 'f1_score' in metric_names
    
    # Check that metadata is included
    for metric in db_metrics:
        assert metric['metadata'] is not None
        metadata = metric['metadata']
        assert 'training_samples' in metadata
        assert 'test_samples' in metadata
        assert 'features' in metadata
        assert 'target_classes' in metadata

def test_model_predictions(sample_data, temp_db):
    """Test that trained model makes reasonable predictions"""
    X_train, X_test, y_train, y_test, feature_names, target_names = sample_data
    
    model = LogisticRegression(random_state=42, max_iter=200)
    
    # Train the model
    trained_model, metrics = train_model_simple(
        model=model,
        model_name="Test_Predictions",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        prediction_logger=temp_db
    )
    
    # Make predictions
    predictions = trained_model.predict(X_test)
    probabilities = trained_model.predict_proba(X_test)
    
    # Check predictions
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1, 2] for pred in predictions)
    
    # Check probabilities
    assert probabilities.shape == (len(X_test), 3)
    assert all(0 <= prob <= 1 for row in probabilities for prob in row)
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in probabilities)

def test_different_model_types(sample_data, temp_db):
    """Test training with different model types"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    X_train, X_test, y_train, y_test, feature_names, target_names = sample_data
    
    models = {
        "Test_Logistic": LogisticRegression(random_state=42, max_iter=200),
        "Test_RandomForest": RandomForestClassifier(n_estimators=10, random_state=42),
        "Test_SVM": SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for model_name, model in models.items():
        trained_model, metrics = train_model_simple(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            target_names=target_names,
            prediction_logger=temp_db
        )
        results[model_name] = metrics
    
    # Check that all models were trained
    assert len(results) == 3
    
    # Check that all models have reasonable performance
    for model_name, metrics in results.items():
        assert metrics['accuracy'] > 0.1  # Very lenient threshold
        assert metrics['f1_score'] > 0.1
    
    # Check database logging
    all_metrics = temp_db.get_training_metrics()
    # Should have 4 metrics per model * 3 models = 12 metrics
    assert len(all_metrics) == 12

def test_metadata_content(sample_data, temp_db):
    """Test that metadata contains expected information"""
    X_train, X_test, y_train, y_test, feature_names, target_names = sample_data
    
    model = LogisticRegression(random_state=42, max_iter=200)
    
    # Train the model
    trained_model, metrics = train_model_simple(
        model=model,
        model_name="Test_Metadata",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        prediction_logger=temp_db
    )
    
    # Get metrics from database
    db_metrics = temp_db.get_training_metrics("Test_Metadata")
    
    # Check metadata content
    for metric in db_metrics:
        metadata = metric['metadata']
        
        # Check required fields
        assert metadata['training_samples'] == len(X_train)
        assert metadata['test_samples'] == len(X_test)
        assert metadata['features'] == X_train.shape[1]
        assert metadata['target_classes'] == list(target_names)
        assert 'model_params' in metadata
        
        # Check model parameters
        model_params = metadata['model_params']
        assert 'random_state' in model_params
        assert model_params['random_state'] == 42

def test_model_performance_reasonable(sample_data, temp_db):
    """Test that model performance is reasonable"""
    X_train, X_test, y_train, y_test, feature_names, target_names = sample_data
    
    model = LogisticRegression(random_state=42, max_iter=200)
    
    # Train the model
    trained_model, metrics = train_model_simple(
        model=model,
        model_name="Test_Performance",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        prediction_logger=temp_db
    )
    
    # Check that performance is better than random guessing (1/3 for 3 classes)
    assert metrics['accuracy'] > 0.4  # Should be better than random
    assert metrics['f1_score'] > 0.4
    
    # Check that precision and recall are reasonable
    assert metrics['precision'] > 0.3
    assert metrics['recall'] > 0.3

def test_consistent_results(sample_data, temp_db):
    """Test that training produces consistent results with same random state"""
    X_train, X_test, y_train, y_test, feature_names, target_names = sample_data
    
    # Train same model twice with same random state
    model1 = LogisticRegression(random_state=42, max_iter=200)
    model2 = LogisticRegression(random_state=42, max_iter=200)
    
    trained_model1, metrics1 = train_model_simple(
        model=model1,
        model_name="Test_Consistent_1",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        prediction_logger=temp_db
    )
    
    trained_model2, metrics2 = train_model_simple(
        model=model2,
        model_name="Test_Consistent_2",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_names=target_names,
        prediction_logger=temp_db
    )
    
    # Results should be identical
    assert abs(metrics1['accuracy'] - metrics2['accuracy']) < 1e-10
    assert abs(metrics1['f1_score'] - metrics2['f1_score']) < 1e-10
