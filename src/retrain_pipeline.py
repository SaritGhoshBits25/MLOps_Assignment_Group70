"""
Model retraining pipeline with data drift detection and automated retraining
"""
import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from database import prediction_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainingPipeline:
    """Pipeline for automated model retraining"""
    
    def __init__(self, 
                 min_samples_for_retrain: int = 100,
                 accuracy_threshold: float = 0.05,
                 data_drift_threshold: float = 0.1):
        self.min_samples_for_retrain = min_samples_for_retrain
        self.accuracy_threshold = accuracy_threshold
        self.data_drift_threshold = data_drift_threshold
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='rbf', random_state=42, probability=True)
        }
    
    def check_retraining_conditions(self) -> dict:
        """Check if model retraining is needed"""
        conditions = {
            'needs_retraining': False,
            'reasons': [],
            'new_data_available': False,
            'data_drift_detected': False,
            'performance_degraded': False,
            'sample_count': 0
        }
        
        try:
            # Get recent predictions from database
            recent_predictions = prediction_logger.get_predictions(limit=1000)
            conditions['sample_count'] = len(recent_predictions)
            
            # Check if we have enough new data
            if len(recent_predictions) >= self.min_samples_for_retrain:
                conditions['new_data_available'] = True
                conditions['reasons'].append(f"Sufficient new data available ({len(recent_predictions)} samples)")
            
            # Check for data drift (simplified - in practice would use more sophisticated methods)
            if self._detect_data_drift(recent_predictions):
                conditions['data_drift_detected'] = True
                conditions['reasons'].append("Data drift detected in recent predictions")
            
            # Check for performance degradation (placeholder - would need ground truth)
            if self._check_performance_degradation():
                conditions['performance_degraded'] = True
                conditions['reasons'].append("Model performance degradation detected")
            
            # Determine if retraining is needed
            conditions['needs_retraining'] = (
                conditions['new_data_available'] or 
                conditions['data_drift_detected'] or 
                conditions['performance_degraded']
            )
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error checking retraining conditions: {str(e)}")
            return conditions
    
    def _detect_data_drift(self, recent_predictions: list) -> bool:
        """Detect data drift in recent predictions (simplified implementation)"""
        try:
            if len(recent_predictions) < 50:
                return False
            
            # Extract input features from recent predictions
            recent_features = []
            for pred in recent_predictions:
                if pred.get('input_data'):
                    features = [
                        pred['input_data']['sepal_length'],
                        pred['input_data']['sepal_width'],
                        pred['input_data']['petal_length'],
                        pred['input_data']['petal_width']
                    ]
                    recent_features.append(features)
            
            if len(recent_features) < 50:
                return False
            
            recent_df = pd.DataFrame(recent_features, columns=[
                'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
            ])
            
            # Load original training data for comparison
            if os.path.exists('data/iris_train.csv'):
                original_df = pd.read_csv('data/iris_train.csv')
                original_features = original_df[['sepal length (cm)', 'sepal width (cm)', 
                                               'petal length (cm)', 'petal width (cm)']]
                
                # Simple drift detection using statistical measures
                for col_recent, col_original in zip(recent_df.columns, original_features.columns):
                    recent_mean = recent_df[col_recent].mean()
                    original_mean = original_features[col_original].mean()
                    
                    # Check if mean has shifted significantly
                    relative_change = abs(recent_mean - original_mean) / original_mean
                    if relative_change > self.data_drift_threshold:
                        logger.info(f"Data drift detected in {col_recent}: {relative_change:.3f}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            return False
    
    def _check_performance_degradation(self) -> bool:
        """Check for model performance degradation (placeholder)"""
        # In a real scenario, this would:
        # 1. Compare recent predictions with ground truth labels
        # 2. Calculate current accuracy/F1 score
        # 3. Compare with baseline performance
        # 4. Return True if performance dropped significantly
        
        # For now, return False as we don't have ground truth for new predictions
        return False
    
    def prepare_retraining_data(self) -> tuple:
        """Prepare data for retraining"""
        try:
            # Get recent predictions
            recent_predictions = prediction_logger.get_predictions(limit=1000)
            
            # Extract features and create synthetic labels based on predictions
            # (In practice, you'd have actual ground truth labels)
            new_features = []
            new_labels = []
            
            for pred in recent_predictions:
                if pred.get('input_data'):
                    features = [
                        pred['input_data']['sepal_length'],
                        pred['input_data']['sepal_width'],
                        pred['input_data']['petal_length'],
                        pred['input_data']['petal_width']
                    ]
                    new_features.append(features)
                    
                    # Map prediction to numeric label
                    label_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
                    new_labels.append(label_map.get(pred['prediction'], 0))
            
            if len(new_features) < self.min_samples_for_retrain:
                raise ValueError(f"Insufficient data for retraining: {len(new_features)} samples")
            
            # Combine with original training data
            if os.path.exists('data/iris_train.csv'):
                original_df = pd.read_csv('data/iris_train.csv')
                original_features = original_df[['sepal length (cm)', 'sepal width (cm)', 
                                               'petal length (cm)', 'petal width (cm)']].values
                original_labels = original_df['target'].values
                
                # Combine datasets
                all_features = np.vstack([original_features, new_features])
                all_labels = np.hstack([original_labels, new_labels])
            else:
                all_features = np.array(new_features)
                all_labels = np.array(new_labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler
            
        except Exception as e:
            logger.error(f"Error preparing retraining data: {str(e)}")
            raise
    
    def retrain_models(self) -> dict:
        """Retrain all models and select the best one"""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test, scaler = self.prepare_retraining_data()
            
            # Set MLflow experiment
            mlflow.set_experiment("iris_retraining")
            
            results = {}
            best_model = None
            best_score = 0
            best_model_name = ""
            
            target_names = ['setosa', 'versicolor', 'virginica']
            
            for model_name, model in self.models.items():
                with mlflow.start_run(run_name=f"retrain_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_param("retrain_timestamp", datetime.now().isoformat())
                    mlflow.log_param("training_samples", len(X_train))
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        model, 
                        "model",
                        registered_model_name=f"iris_{model_name.lower().replace(' ', '_')}_retrained"
                    )
                    
                    results[model_name] = {
                        'accuracy': accuracy,
                        'f1_score': f1
                    }
                    
                    # Track best model
                    if f1 > best_score:
                        best_score = f1
                        best_model = model
                        best_model_name = model_name
                    
                    logger.info(f"Retrained {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Save best retrained model
            if best_model:
                # Create backup of current model
                if os.path.exists('models/best_model.pkl'):
                    backup_path = f"models/best_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    os.rename('models/best_model.pkl', backup_path)
                
                # Save new best model
                joblib.dump(best_model, 'models/best_model.pkl')
                joblib.dump(scaler, 'models/scaler.pkl')
                
                # Update model info
                model_info = {
                    'best_model': best_model_name,
                    'best_score': best_score,
                    'feature_names': ['sepal length (cm)', 'sepal width (cm)', 
                                    'petal length (cm)', 'petal width (cm)'],
                    'target_names': target_names,
                    'model_version': f"2.1.{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'retrain_timestamp': datetime.now().isoformat(),
                    'training_samples': len(X_train)
                }
                
                with open('models/model_info.json', 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                logger.info(f"Model retraining completed. New best model: {best_model_name} (F1: {best_score:.4f})")
            
            return {
                'success': True,
                'best_model': best_model_name,
                'best_score': best_score,
                'all_results': results,
                'training_samples': len(X_train),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during model retraining: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_retraining_check(self) -> dict:
        """Run complete retraining check and execute if needed"""
        logger.info("Starting retraining check...")
        
        # Check conditions
        conditions = self.check_retraining_conditions()
        
        result = {
            'retraining_triggered': False,
            'conditions': conditions,
            'timestamp': datetime.now().isoformat()
        }
        
        if conditions['needs_retraining']:
            logger.info("Retraining conditions met. Starting retraining...")
            retrain_result = self.retrain_models()
            result['retraining_triggered'] = True
            result['retrain_result'] = retrain_result
        else:
            logger.info("Retraining not needed at this time.")
        
        return result

def main():
    """Main function for running retraining pipeline"""
    pipeline = ModelRetrainingPipeline()
    result = pipeline.run_retraining_check()
    
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()
