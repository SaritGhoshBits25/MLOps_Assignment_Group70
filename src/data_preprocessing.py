"""
Data preprocessing module for Iris dataset
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Load and preprocess the Iris dataset
    
    Args:
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, target_names)
    """
    logger.info("Loading Iris dataset...")
    
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create DataFrame for better handling
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = [target_names[i] for i in y]
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Target classes: {target_names}")
    logger.info(f"Class distribution:\n{df['target_name'].value_counts()}")
    
    # Save raw data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/iris_raw.csv', index=False)
    logger.info("Raw data saved to data/iris_raw.csv")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training set size: {X_train_scaled.shape[0]}")
    logger.info(f"Test set size: {X_test_scaled.shape[0]}")
    
    # Save processed data
    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df['target'] = y_train
    train_df.to_csv('data/iris_train.csv', index=False)
    
    test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    test_df['target'] = y_test
    test_df.to_csv('data/iris_test.csv', index=False)
    
    logger.info("Processed data saved to data/iris_train.csv and data/iris_test.csv")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names, scaler

def get_sample_input():
    """
    Get a sample input for testing the API
    
    Returns:
        dict: Sample input data
    """
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

if __name__ == "__main__":
    # Test the preprocessing
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
    print("Data preprocessing completed successfully!")
