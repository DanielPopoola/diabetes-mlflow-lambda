import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"

DATASET_PATH = DATA_DIR / "raw" / "diabetes.csv"
TRAIN_TEST_SPLIT_CONFIG = {
    "test_size": 0.3,
    "random_state": 0,
    "stratify": True  # Maintain class balance
}

NUMERICAL_FEATURES = [0, 1, 2, 3, 4, 5, 6]
CATEGORICAL_FEATURES = [7]
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": 10,
        "min_samples_split": 5
    }
}

MLFLOW_EXPERIMENT_NAME = "diabetes_classification"
MODEL_NAME = "RandomForestClassifier"

LAMBDA_PACKAGE_SIZE_LIMIT = 50 * 1024 * 1024
print(MODELS_DIR)