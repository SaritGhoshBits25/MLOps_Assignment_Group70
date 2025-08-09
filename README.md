# MLOps Iris Classification API - Group 70

A comprehensive MLOps implementation for Iris flower classification with automated training, monitoring, and deployment capabilities.

## Features

- **Machine Learning Models**: Multiple algorithms (Logistic Regression, Random Forest, SVM)
- **FastAPI REST API**: High-performance API with automatic documentation
- **MLflow Integration**: Experiment tracking and model versioning
- **Prometheus Monitoring**: Real-time metrics and performance monitoring
- **Docker Containerization**: Easy deployment and scaling
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Data Version Control (DVC)**: Track data and model versions
- **Comprehensive Testing**: Unit tests with pytest and coverage reporting
- **Database Logging**: SQLite database for prediction logging
- **Health Checks**: API health monitoring and status endpoints

## Architecture

```
├── src/                    # Source code
│   ├── api.py             # FastAPI application with monitoring
│   ├── train.py           # Model training with MLflow
│   ├── data_preprocessing.py # Data preprocessing pipeline
│   ├── database.py        # Database operations and logging
│   └── retrain_pipeline.py # Automated retraining pipeline
├── data/                  # Dataset files
│   ├── iris_raw.csv       # Raw iris dataset
│   ├── iris_train.csv     # Training data
│   └── iris_test.csv      # Test data
├── models/                # Trained model artifacts
├── tests/                 # Test suite
├── monitoring/            # Monitoring configuration
│   └── prometheus.yml     # Prometheus configuration
├── .github/workflows/     # CI/CD pipeline
│   └── ci-cd.yml         # GitHub Actions workflow
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service orchestration
└── requirements.txt      # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLOps_Assignment_Group70-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess data**
   ```bash
   python src/data_preprocessing.py
   ```

4. **Train models**
   ```bash
   python src/train.py
   ```

5. **Start the API**
   ```bash
   python src/api.py
   ```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

This will start:
- **Iris API**: `http://localhost:8000`
- **Prometheus**: `http://localhost:9090`
- **MLflow**: `http://localhost:5000`

## API Endpoints

### Core Endpoints

- `GET /` - API information and status
- `GET /health` - Health check endpoint
- `POST /predict` - Make predictions
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Current model information
- `GET /metrics` - Prometheus metrics

### Example Usage

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "instances": [
         {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
         {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}
       ]
     }'
```

## Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test files
pytest tests/test_api.py -v
pytest tests/test_training.py -v
```

## Monitoring and Observability

### Prometheus Metrics

The API exposes various metrics:
- `predictions_total` - Total number of predictions
- `prediction_duration_seconds` - Prediction latency
- `api_requests_total` - Total API requests
- `model_accuracy` - Current model accuracy
- `active_models` - Number of active models

### MLflow Tracking

- Experiment tracking with model parameters and metrics
- Model versioning and artifact storage
- Model comparison and selection
- Access MLflow UI at `http://localhost:5000`

### Database Logging

All predictions are logged to SQLite database with:
- Prediction ID and timestamp
- Input features and predictions
- Model version and confidence scores
- Request metadata

## CI/CD Pipeline

The GitHub Actions workflow includes:

1. **Testing Stage**
   - Code quality checks
   - Unit test execution
   - API health verification

2. **Build Stage**
   - Docker image building
   - Container testing
   - Image registry push

3. **Deploy Stage**
   - Production deployment (configurable)
   - Health monitoring

## Development

### Adding New Models

1. Implement model in `src/train.py`
2. Add model configuration
3. Update API endpoints if needed
4. Add corresponding tests

### Data Pipeline

The data preprocessing pipeline:
1. Loads raw iris dataset
2. Performs data validation and cleaning
3. Splits into train/test sets
4. Saves processed data for training

### Model Retraining

Automated retraining pipeline (`src/retrain_pipeline.py`):
- Monitors model performance
- Triggers retraining based on thresholds
- Updates model artifacts
- Logs retraining metrics

## 📊 Data Version Control (DVC)

This project uses DVC for data versioning and pipeline management. DVC tracks data files and ensures reproducible data processing workflows.

### DVC Setup

The project is configured with:
- **Data tracking**: Raw iris dataset (`data/iris_raw.csv`)
- **Pipeline**: Automated data preprocessing pipeline
- **Remote storage**: Local remote for data versioning

### DVC Files Structure

```
├── .dvc/
│   └── config              # DVC configuration
├── data/
│   ├── .gitignore         # Git ignores data files
│   ├── iris_raw.csv.dvc   # DVC tracks raw data
│   ├── iris_train.csv     # Generated by pipeline
│   └── iris_test.csv      # Generated by pipeline
├── dvc.yaml               # Pipeline definition
└── dvc.lock               # Pipeline lock file
```

### DVC Commands

**Check pipeline status:**
```bash
dvc status
```

**Run the data preprocessing pipeline:**
```bash
dvc repro
```

**Pull data from remote:**
```bash
dvc pull
```

**Push data to remote:**
```bash
dvc push
```

**Show pipeline DAG:**
```bash
dvc dag
```

### Data Pipeline

The DVC pipeline includes:

1. **Data Preprocessing Stage**
   - **Input**: `data/iris_raw.csv`
   - **Output**: `data/iris_train.csv`, `data/iris_test.csv`
   - **Command**: `python3 src/data_preprocessing.py`

### Working with Data

**To modify the dataset:**
1. Update `data/iris_raw.csv`
2. Run `dvc repro` to regenerate processed data
3. Commit changes: `git add dvc.lock data/iris_raw.csv.dvc`

**To add new data files:**
```bash
dvc add data/new_dataset.csv
git add data/new_dataset.csv.dvc data/.gitignore
```

### Remote Storage

The project uses a local remote storage. For production, you can configure cloud storage:

**AWS S3:**
```bash
dvc remote add -d myremote s3://mybucket/dvcstore
```

**Google Cloud Storage:**
```bash
dvc remote add -d myremote gs://mybucket/dvcstore
```

**Azure Blob Storage:**
```bash
dvc remote add -d myremote azure://mycontainer/dvcstore
```

### Benefits of DVC Integration

- **Data Versioning**: Track changes to datasets over time
- **Reproducibility**: Ensure consistent data processing
- **Collaboration**: Share data efficiently across team members
- **Pipeline Management**: Automate data processing workflows
- **Storage Efficiency**: Avoid storing large files in Git

## Configuration

### Environment Variables

- `PYTHONPATH`: Python path configuration
- `MODEL_PATH`: Path to model artifacts
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

### Docker Configuration

- **Dockerfile**: Multi-stage build for optimized images
- **docker-compose.yml**: Multi-service orchestration
- Health checks and restart policies included

### Logs

- API logs: `logs/api.log`
- Container logs: `docker-compose logs <service-name>`
- MLflow logs: Available in MLflow UI

## Team Members

1. Dhiman Kundu
2. Rina Gupta
3. Sarit Ghosh
4. Soumen Choudhury

## License

This project is part of an MLOps assignment for Group 70.

## Model Information

- **Dataset**: Iris flower classification
- **Features**: Sepal length/width, Petal length/width
- **Classes**: Setosa, Versicolor, Virginica
- **Models**: Logistic Regression, Random Forest, SVM
- **Evaluation**: Accuracy, Precision, Recall, F1-score


**Group 70 MLOps Assignment** - A comprehensive machine learning operations implementation with modern DevOps practices.
