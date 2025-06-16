import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import os
from pathlib import Path
from .preprocessing import load_and_preprocess_data
from ..config import MODEL_CONFIG, MLFLOW_EXPERIMENT_NAME, MODEL_NAME, MODELS_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def create_processor(numeric_features, categorical_features):
    """Create the processing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def train_model(environment="local"):
    """Train RandomForest model with MLflow tracking."""

    if environment == "docker":
        mlflow.set_tracking_uri("file:///app/mlflow/docker/mlflow.db")
    else:
        mlflow.set_tracking_uri("file:///app/mlflow/local/mlflow.db")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

    with mlflow.start_run() as run:
        params = MODEL_CONFIG["random_forest"]


        mlflow.log_params(params)
        mlflow.log_param("environment", environment)
        mlflow.log_param("python_version", "3.9")
        mlflow.log_param("feature_count", len(feature_names))

        preprocessor = create_processor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)


        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('randomforest', RandomForestClassifier(**params))])  # type: ignore

        model = pipeline.fit(X_train, y_train)

        test_pred = model.predict(X_test)
        test_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = get_metrics(y_test, test_pred, test_pred_proba)

        input_example = X_test[:5]
        signature = infer_signature(X_test, test_pred)

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
            metadata={
                "environment": environment,
                "python_version": "3.9",
                "deployment_ready": environment == 'docker'
            }
        )

        # Save model locally for validation
        model_dir = MODELS_DIR / environment
        model_dir.mkdir(exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")

        print(f"Model trained in {environment} environment:")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Test Accuracy: {metrics['accuracy']}")
        print(f"  Test AUC: {metrics['auc']}")
        
        return run.info.run_id, model



def get_metrics(y_test, y_pred, y_pred_proba):
    """Get relevant metrics to evaluate model performance."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_score = str(roc_auc_score(y_test, y_pred_proba))

    return {'accuracy': round(float(accuracy), 2), 
            'precision': round(float(precision), 2), 
            'recall': round(float(recall), 2), 
            'auc': auc_score
        }

if __name__ == "__main__":
    import sys
    env = sys.argv[1] if len(sys.argv) > 1 else "local"
    train_model(environment=env)