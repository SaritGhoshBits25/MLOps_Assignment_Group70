"""
Database models and operations for logging and monitoring
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionLogger:
    """Class to handle prediction logging to SQLite database"""
    
    def __init__(self, db_path: str = "logs/predictions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        request_id TEXT,
                        input_data TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        probability REAL NOT NULL,
                        model_version TEXT NOT NULL,
                        processing_time_ms REAL,
                        client_ip TEXT,
                        user_agent TEXT
                    )
                """)
                
                # Create model_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # Create api_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        method TEXT NOT NULL,
                        status_code INTEGER NOT NULL,
                        response_time_ms REAL NOT NULL,
                        client_ip TEXT,
                        error_message TEXT
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def log_prediction(self, 
                      input_data: Dict[str, Any],
                      prediction: str,
                      probability: float,
                      model_version: str,
                      processing_time_ms: float,
                      request_id: str = None,
                      client_ip: str = None,
                      user_agent: str = None):
        """Log a prediction to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO predictions 
                    (timestamp, request_id, input_data, prediction, probability, 
                     model_version, processing_time_ms, client_ip, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    request_id,
                    json.dumps(input_data),
                    prediction,
                    probability,
                    model_version,
                    processing_time_ms,
                    client_ip,
                    user_agent
                ))
                conn.commit()
                logger.info(f"Prediction logged: {prediction} (prob: {probability:.4f})")
                
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
    
    def log_model_metric(self, 
                        model_name: str,
                        metric_name: str,
                        metric_value: float,
                        metadata: Dict[str, Any] = None):
        """Log model metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO model_metrics 
                    (timestamp, model_name, metric_name, metric_value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    model_name,
                    metric_name,
                    metric_value,
                    json.dumps(metadata) if metadata else None
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging model metric: {str(e)}")
    
    def log_api_metric(self,
                      endpoint: str,
                      method: str,
                      status_code: int,
                      response_time_ms: float,
                      client_ip: str = None,
                      error_message: str = None):
        """Log API metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO api_metrics 
                    (timestamp, endpoint, method, status_code, response_time_ms, client_ip, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    endpoint,
                    method,
                    status_code,
                    response_time_ms,
                    client_ip,
                    error_message
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging API metric: {str(e)}")
    
    def get_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    # Parse JSON fields
                    if result['input_data']:
                        result['input_data'] = json.loads(result['input_data'])
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return []
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total predictions
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total_predictions = cursor.fetchone()[0]
                
                # Predictions by class
                cursor.execute("""
                    SELECT prediction, COUNT(*) as count 
                    FROM predictions 
                    GROUP BY prediction
                """)
                predictions_by_class = dict(cursor.fetchall())
                
                # Average processing time
                cursor.execute("SELECT AVG(processing_time_ms) FROM predictions")
                avg_processing_time = cursor.fetchone()[0] or 0
                
                # Recent predictions (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE datetime(timestamp) > datetime('now', '-1 day')
                """)
                recent_predictions = cursor.fetchone()[0]
                
                return {
                    'total_predictions': total_predictions,
                    'predictions_by_class': predictions_by_class,
                    'avg_processing_time_ms': round(avg_processing_time, 2),
                    'recent_predictions_24h': recent_predictions
                }
                
        except Exception as e:
            logger.error(f"Error getting prediction stats: {str(e)}")
            return {}

    def get_training_metrics(self, model_name: str = None) -> List[Dict[str, Any]]:
        """Get training metrics for a specific model or all models"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if model_name:
                    cursor.execute("""
                        SELECT * FROM model_metrics 
                        WHERE model_name = ?
                        ORDER BY timestamp DESC
                    """, (model_name,))
                else:
                    cursor.execute("""
                        SELECT * FROM model_metrics 
                        ORDER BY timestamp DESC
                    """)
                
                columns = [description[0] for description in cursor.description]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    # Parse JSON metadata
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting training metrics: {str(e)}")
            return []
    
    def get_best_model_info(self) -> Dict[str, Any]:
        """Get information about the best model selection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM model_metrics 
                    WHERE metric_name = 'model_selection'
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    result = dict(zip(columns, row))
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    return result
                
                return {}
                
        except Exception as e:
            logger.error(f"Error getting best model info: {str(e)}")
            return {}

# Global logger instance
prediction_logger = PredictionLogger()
