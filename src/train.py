"""
Model training module with MLflow experiment tracking and database logging
"""
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
import logging
from datetime import datetime
from data_preprocessing import load_and_preprocess_data
from database import prediction_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(model, model_name, X_train, X_test, y_train, y_test, target_names):
    """
    Train a model and log metrics to MLflow and SQLite database
    
    Args:
        model: Sklearn model instance
        model_name (str): Name of the model
        X_train, X_test, y_train, y_test: Training and test data
        target_names: Names of target classes
        
    Returns:
        tuple: (trained_model, metrics)
    """
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Train the model
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log metrics to SQLite database
        training_metadata = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(X_train.columns) if hasattr(X_train, 'columns') else X_train.shape[1],
            'model_params': model.get_params(),
            'target_classes': list(target_names)
        }
        
        prediction_logger.log_model_metric(model_name, 'accuracy', accuracy, training_metadata)
        prediction_logger.log_model_metric(model_name, 'precision', precision, training_metadata)
        prediction_logger.log_model_metric(model_name, 'recall', recall, training_metadata)
        prediction_logger.log_model_metric(model_name, 'f1_score', f1, training_metadata)
        
        logger.info(f"Logged training metrics to database for {model_name}")
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"iris_{model_name.lower().replace(' ', '_')}"
        )
        
        # Save model locally
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{model_name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_path)
        
        # Log classification report
        report = classification_report(y_test, y_pred, target_names=target_names)
        logger.info(f"\n{model_name} Classification Report:\n{report}")
        
        # Log as artifact
        with open(f"models/{model_name.lower().replace(' ', '_')}_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact(f"models/{model_name.lower().replace(' ', '_')}_report.txt")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return model, metrics

def main():
    """
    Main training function
    """
    # Set MLflow experiment
    mlflow.set_experiment("iris_classification")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
    
    # Save scaler for later use in API
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    # Train all models and track results
    results = {}
    best_model = None
    best_score = 0
    best_model_name = ""
    
    logger.info("Starting model training and logging to database...")
    
    for model_name, model in models.items():
        trained_model, metrics = train_model(
            model, model_name, X_train, X_test, y_train, y_test, target_names
        )
        results[model_name] = metrics
        
        # Track best model
        if metrics['f1_score'] > best_score:
            best_score = metrics['f1_score']
            best_model = trained_model
            best_model_name = model_name
    
    # Save best model
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Log best model selection to database
    best_model_metadata = {
        'selected_from_models': list(models.keys()),
        'selection_criteria': 'f1_score',
        'all_model_scores': {name: results[name]['f1_score'] for name in results},
        'training_timestamp': datetime.now().isoformat(),
        'model_version': '1.0.0'
    }
    
    prediction_logger.log_model_metric(
        f"BEST_MODEL_{best_model_name}", 
        'model_selection', 
        best_score, 
        best_model_metadata
    )
    
    # Log training summary
    training_summary = {
        'total_models_trained': len(models),
        'best_model': best_model_name,
        'best_f1_score': best_score,
        'training_data_size': len(X_train),
        'test_data_size': len(X_test),
        'feature_count': len(feature_names),
        'target_classes': list(target_names)
    }
    
    prediction_logger.log_model_metric(
        'TRAINING_SUMMARY', 
        'training_completed', 
        1.0, 
        training_summary
    )
    
    # Log best model info
    logger.info(f"\nBest Model: {best_model_name} (F1 Score: {best_score:.4f})")
    logger.info("All training metrics logged to SQLite database!")
    
    # Save model metadata
    model_info = {
        'best_model': best_model_name,
        'best_score': best_score,
        'feature_names': list(feature_names),
        'target_names': list(target_names),
        'model_version': '1.0.0'
    }
    
    import json
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("Training completed successfully!")
    return results

if __name__ == "__main__":
    results = main()
