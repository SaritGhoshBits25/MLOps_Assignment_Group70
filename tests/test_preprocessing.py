"""
Unit tests for data preprocessing
"""
import pytest
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_and_preprocess_data, get_sample_input

def test_load_and_preprocess_data():
    """Test data loading and preprocessing"""
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
    
    # Check shapes
    assert X_train.shape[1] == 4  # 4 features
    assert X_test.shape[1] == 4
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    
    # Check that we have reasonable train/test split
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[0] + X_test.shape[0] == 150  # Total iris samples
    
    # Check feature names
    assert len(feature_names) == 4
    assert 'sepal length' in feature_names[0]
    
    # Check target names
    assert len(target_names) == 3
    assert 'setosa' in target_names
    assert 'versicolor' in target_names
    assert 'virginica' in target_names
    
    # Check that data is scaled (mean should be close to 0)
    assert abs(np.mean(X_train)) < 0.5  # Relaxed threshold
    
    # Check that scaler is fitted
    assert hasattr(scaler, 'mean_')
    assert hasattr(scaler, 'scale_')

def test_get_sample_input():
    """Test sample input generation"""
    sample = get_sample_input()
    
    assert isinstance(sample, dict)
    assert 'sepal_length' in sample
    assert 'sepal_width' in sample
    assert 'petal_length' in sample
    assert 'petal_width' in sample
    
    # Check that all values are positive
    for key, value in sample.items():
        assert value > 0
        assert isinstance(value, (int, float))

def test_data_files_created():
    """Test that data files are created"""
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Run preprocessing
    load_and_preprocess_data()
    
    # Check that files exist
    assert os.path.exists('data/iris_raw.csv')
    assert os.path.exists('data/iris_train.csv')
    assert os.path.exists('data/iris_test.csv')
    
    # Check file contents
    raw_df = pd.read_csv('data/iris_raw.csv')
    assert raw_df.shape[0] == 150  # 150 iris samples
    assert raw_df.shape[1] == 6  # 4 features + target + target_name
    assert 'target' in raw_df.columns
    assert 'target_name' in raw_df.columns
    
    # Check train/test files
    train_df = pd.read_csv('data/iris_train.csv')
    test_df = pd.read_csv('data/iris_test.csv')
    
    assert train_df.shape[0] > 0
    assert test_df.shape[0] > 0
    assert train_df.shape[0] + test_df.shape[0] == 150

def test_feature_scaling():
    """Test that features are properly scaled"""
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
    
    # Check that training data is scaled
    train_means = np.mean(X_train, axis=0)
    train_stds = np.std(X_train, axis=0)
    
    # Means should be close to 0
    assert all(abs(mean) < 0.1 for mean in train_means)
    
    # Standard deviations should be close to 1
    assert all(abs(std - 1.0) < 0.1 for std in train_stds)
    
    # Test data should also be scaled using the same scaler
    test_means = np.mean(X_test, axis=0)
    # Test means might not be exactly 0 due to different distribution
    assert all(abs(mean) < 2.0 for mean in test_means)

def test_target_encoding():
    """Test that targets are properly encoded"""
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_and_preprocess_data()
    
    # Check that targets are integers
    assert all(isinstance(y, (int, np.integer)) for y in y_train)
    assert all(isinstance(y, (int, np.integer)) for y in y_test)
    
    # Check that target values are in expected range
    all_targets = np.concatenate([y_train, y_test])
    unique_targets = np.unique(all_targets)
    
    assert len(unique_targets) == 3  # 3 classes
    assert set(unique_targets) == {0, 1, 2}  # Should be 0, 1, 2
