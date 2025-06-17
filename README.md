# Diabetes Prediction: Serverless ML Deployment

A production-ready machine learning system for diabetes prediction using MLflow model registry and AWS Lambda deployment. This project demonstrates best practices for reproducible ML workflows, cross-environment validation, and serverless deployment.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local Dev     │    │   Docker Env    │    │   AWS Lambda    │
│                 │    │                 │    │                 │
│ • Training      │───▶│ • Training      │───▶│ • Inference     │
│ • Experimentation│    │ • Validation    │    │ • API Gateway   │
│ • MLflow UI     │    │ • Package Build │    │ • Production    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   MLflow Registry│
                    │                  │
                    │ • Model Tracking │
                    │ • Version Control│
                    │ • Stage Management│
                    └──────────────────┘
```

## 🎯 Key Features

- **Cross-Environment Consistency**: Identical model behavior across local, Docker, and Lambda environments
- **MLflow Integration**: Complete model lifecycle management with experiment tracking and registry
- **Automated Validation**: Comprehensive cross-environment model validation pipeline  
- **Production-Ready**: Lambda deployment package with proper error handling and monitoring
- **Reproducible Workflows**: Pinned dependencies and consistent Python environments
- **Scalable Architecture**: Serverless deployment for cost-effective inference

## 📁 Project Structure

```
diabetes-mlflow-lambda/
├── src/
│   ├── config.py                    # Project configuration
│   ├── models/
│   │   ├── preprocessing.py         # Data preprocessing pipeline
│   │   └── training.py             # Model training with MLflow
│   ├── deployment/
│   │   └── lambda_handler.py       # AWS Lambda inference handler
│   └── validation/
│       └── cross_env_validation.py # Cross-environment testing
├── data/
│   ├── raw/                        # Original dataset
│   └── splits/                     # Train/test splits and metadata
├── mlflow/
│   ├── local/                      # Local MLflow tracking
│   └── docker/                     # Docker MLflow tracking
├── models/
│   ├── local/                      # Local trained models
│   └── production/                 # Production-ready models
├── lambda_package/                 # Lambda deployment artifacts
├── scripts/
│   └── build_lambda_package.py     # Package builder
├── environments/
│   └── docker/
│       ├── Dockerfile              # Lambda-compatible environment
│       └── docker-compose.yml      # Container orchestration
├── requirements.txt                # Python dependencies
└── pyproject.toml                  # UV project configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9
- [UV](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- AWS CLI (for deployment)

### 1. Environment Setup

```bash
# Clone and setup project
git clone https://github.com/DanielPopoola/diabetes-mlflow-lambda.git
cd diabetes-mlflow-lambda

# Initialize with UV
uv sync

# Create directory structure
mkdir -p {data/raw,mlflow/{local,docker},models/{local,production}}
```

### 2. Data Preparation

```bash
# Place your diabetes dataset
cp /path/to/your/diabetes.csv data/raw/

# Verify data structure
head data/raw/diabetes.csv
```

Expected columns: `Pregnancies`, `PlasmaGlucose`, `DiastolicBloodPressure`, `TricepsThickness`, `SerumInsulin`, `BMI`, `DiabetesPedigree`, `Age`, `Diabetic`

### 3. Local Development

```bash
# Train model locally
uv run python -m src.models.training local

# View MLflow UI
uv run mlflow ui --backend-store-uri file://$(pwd)/mlflow/local/mlflow.db --port 5000
```

Visit http://localhost:5000 to explore experiments and models.

### 4. Docker Environment Training

```bash
# Build and start Docker environment
cd environments/docker
docker-compose build
docker-compose up -d

# Enter container and train
docker-compose exec lambda-training bash
python -m src.models.training docker

# Promote model to Production stage
python -c "
import mlflow
mlflow.set_tracking_uri('file:///app/mlflow/docker/mlflow.db')
client = mlflow.MlflowClient()
latest_version = client.get_latest_versions('RandomForestClassifier')[0]
client.transition_model_version_stage('RandomForestClassifier', latest_version.version, 'Production')
print(f'Model version {latest_version.version} promoted to Production')
"
```

### 5. Cross-Environment Validation

```bash
# Copy Docker model for validation
docker cp diabetes-lambda-env:/app/models/docker/model.joblib models/production/

# Run comprehensive validation
uv run python -m src.validation.cross_env_validation
```

Expected output: All validation tests should pass ✅

### 6. Lambda Package Creation

```bash
# Inside Docker container
python scripts/build_lambda_package.py

# Verify package size (should be < 50MB)
ls -lh lambda_package/deployment.zip
```

### 7. Local Testing

```bash
# Test Lambda handler locally
cd lambda_package
python lambda_function.py
```

## 🧪 Model Validation

The cross-environment validation system ensures model consistency across environments by testing:

- **Parameter Consistency**: Model hyperparameters match exactly  
- **Prediction Consistency**: Binary predictions are identical
- **Probability Consistency**: Prediction probabilities match within tolerance
- **Performance Metrics**: Accuracy and AUC scores are consistent
- **Edge Case Handling**: Robust behavior on boundary conditions

## 📊 Model Performance

Current model performance on test set:
- **Accuracy**: ~93%
- **AUC-ROC**: ~0.98

## 🔧 Configuration

Key configuration options in `src/config.py`:

```python
# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": 10,
        "min_samples_split": 5
    }
}

# Train/test split
TRAIN_TEST_SPLIT_CONFIG = {
    "test_size": 0.3,
    "random_state": 0,
    "stratify": True
}
```

## 🚀 AWS Deployment

### Lambda Function Setup

1. **Create Lambda Function**:
   ```bash
   aws lambda create-function \
     --function-name diabetes-prediction \
     --runtime python3.9 \
     --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
     --handler lambda_function.lambda_handler \
     --zip-file fileb://lambda_package/deployment.zip
   ```

2. **Configure Memory and Timeout**:
   ```bash
   aws lambda update-function-configuration \
     --function-name diabetes-prediction \
     --memory-size 512 \
     --timeout 30
   ```

### API Gateway Integration

Create REST API endpoint:

```bash
# Example API call
curl -X POST https://your-api-gateway-url/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "PlasmaGlucose": 148,
    "DiastolicBloodPressure": 72,
    "TricepsThickness": 35,
    "SerumInsulin": 0,
    "BMI": 33.6,
    "DiabetesPedigree": 0.627,
    "Age": 50
  }'
```

Expected response:
```json
{
  "prediction": 1,
  "probability": {
    "non_diabetic": 0.482,
    "diabetic": 0.518
  },
  "confidence": 0.75,
  "risk_level": "high"
}
```

## 🔍 Monitoring and Observability

### MLflow Tracking

- **Experiments**: All training runs tracked with parameters and metrics
- **Model Registry**: Versioned models with stage management
- **Artifacts**: Model binaries and preprocessing objects stored

### Lambda Monitoring

- **CloudWatch Logs**: Function execution logs
- **CloudWatch Metrics**: Invocation count, duration, errors
- **Custom Metrics**: Prediction confidence, feature distributions

## 🛠️ Development Workflow

### Adding New Features

1. **Feature Engineering**: Add preprocessing in `src/models/preprocessing.py`
2. **Model Updates**: Modify training in `src/models/training.py`  
3. **Validation**: Update validation tests in `src/validation/cross_env_validation.py`
4. **Testing**: Run full validation pipeline
5. **Deployment**: Rebuild Lambda package

### Model Retraining

```bash
# Local training
uv run python -m src.models.training local

# Docker training  
docker-compose exec lambda-training python -m src.models.training docker

# Validate consistency
uv run python -m src.validation.cross_env_validation

# Promote to production
# (MLflow model stage transition)
```

## 📈 Performance Optimization

### Model Size Optimization

- **Feature Selection**: Remove low-importance features
- **Model Compression**: Use smaller ensemble sizes
- **Quantization**: Reduce model precision for inference

### Lambda Cold Start Optimization

- **Provisioned Concurrency**: Keep functions warm
- **Lazy Loading**: Load model components on demand
- **Memory Optimization**: Right-size Lambda memory allocation

## 🔒 Security Considerations

- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: No sensitive information in error messages
- **IAM Roles**: Principle of least privilege for Lambda execution
- **API Gateway**: Rate limiting and authentication

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues:

- **Documentation**: Check this README and inline code comments
- **Issues**: Open a GitHub issue with detailed description
- **MLflow UI**: Use local MLflow interface for experiment exploration
- **Validation**: Run cross-environment validation for debugging

## 🎯 Roadmap

- [ ] **Multi-model Support**: Compare different algorithms
- [ ] **Feature Store Integration**: Centralized feature management  
- [ ] **A/B Testing**: Model comparison in production
- [ ] **Real-time Monitoring**: Model drift detection
- [ ] **Automated Retraining**: Scheduled model updates
- [ ] **Multi-cloud Deployment**: Support for GCP Cloud Functions, Azure Functions

---

**Built with ❤️ for production-ready ML deployments**
