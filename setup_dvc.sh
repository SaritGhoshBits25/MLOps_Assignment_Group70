#!/bin/bash

# DVC Setup Script for MLOps Iris Classification Project
# This script sets up DVC for data versioning

echo "Setting up DVC for data versioning..."

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "DVC is not installed. Installing DVC..."
    pip install dvc
fi

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
fi

# Add data to DVC tracking
echo "Adding data to DVC tracking..."
if [ -f "data/iris_raw.csv" ]; then
    dvc add data/iris_raw.csv
    echo "Raw data added to DVC tracking"
else
    echo "Warning: data/iris_raw.csv not found"
fi

# Set up local remote storage
echo "Setting up local remote storage..."
mkdir -p ../dvc-storage
dvc remote add -d myremote ../dvc-storage

# Run the data preprocessing pipeline
echo "Running data preprocessing pipeline..."
dvc repro

# Show status
echo "DVC setup completed!"
echo "Current DVC status:"
dvc status

echo ""
echo "Next steps:"
echo "1. Run 'dvc repro' to execute the data pipeline"
echo "2. Run 'dvc push' to push data to remote storage"
echo "3. Run 'dvc pull' to pull data from remote storage"
echo "4. Add DVC files to git: git add .dvc/config *.dvc dvc.yaml dvc.lock"
