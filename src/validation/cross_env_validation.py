import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path
from ..config import MODELS_DIR, MODEL_NAME
from ..models.preprocessing import load_and_preprocess_data


def load_models():
    """Load models from both environments."""
    local_model = joblib.load(MODELS_DIR / "local" / "model.joblib")
    docker_model = joblib.load(MODELS_DIR / "production" / "model.joblib")
    return local_model, docker_model

def validate_model_consistency(tolerance=1e-5):
    """Validate that models produce consistent results."""

    print("🔍 Starting cross-environment model validation...")

    try:
        local_model, docker_model = load_models()
        print("✅ Models loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

    # Test 1: Model parameters consistency
    print("\n📊 Test 1: Model Parameters")
    local_params = local_model.get_params()
    docker_params = docker_model.get_params()
    
    params_match = local_params == docker_params
    print(f"  Parameters match: {'✅' if params_match else '❌'}")

    if not params_match:
        print("  Parameter differences:")
        for key in local_params:
            if local_params[key] != docker_params[key]:
                print(f"    {key}: {local_params[key]} vs {docker_params[key]}")

    # Test 2: Prediction consistency
    print("\n🔮 Test 2: Prediction Consistency")
    local_pred = local_model.predict(X_test)
    docker_pred = docker_model.predict(X_test)
    
    predictions_match = np.array_equal(local_pred, docker_pred)
    print(f"  Binary predictions match: {'✅' if predictions_match else '❌'}")
    
    if not predictions_match:
        diff_count = np.sum(local_pred != docker_pred)
        print(f"  Different predictions: {diff_count}/{len(local_pred)} ({diff_count/len(local_pred)*100:.2f}%)")

    # Test 3: Probability consistency
    print("\n📈 Test 3: Probability Consistency")
    local_proba = local_model.predict_proba(X_test)
    docker_proba = docker_model.predict_proba(X_test)
    
    proba_close = np.allclose(local_proba, docker_proba, rtol=tolerance)
    print(f"  Probabilities match: {'✅' if proba_close else '❌'}")
    
    if not proba_close:
        proba_diff = np.abs(local_proba - docker_proba)
        max_diff = np.max(proba_diff)
        print(f"  Max probability difference: {max_diff:.8f}")
    
    # Test 4: Performance metrics
    print("\n📊 Test 4: Performance Metrics")
    local_accuracy = accuracy_score(y_test, local_pred)
    docker_accuracy = accuracy_score(y_test, docker_pred)
    local_auc = roc_auc_score(y_test, local_proba[:, 1])
    docker_auc = roc_auc_score(y_test, docker_proba[:, 1])
    
    print(f"  Local - Accuracy: {local_accuracy:.6f}, AUC: {local_auc:.6f}")
    print(f"  Docker - Accuracy: {docker_accuracy:.6f}, AUC: {docker_auc:.6f}")

    accuracy_close = abs(local_accuracy - docker_accuracy) < tolerance
    auc_close = abs(local_auc - docker_auc) < tolerance
    
    print(f"  Accuracy match: {'✅' if accuracy_close else '❌'}")
    print(f"  AUC match: {'✅' if auc_close else '❌'}")
    
    # Test 6: Edge cases
    print("\n⚠️ Test 6: Edge Cases")
    edge_cases = create_edge_cases(feature_names)
    
    local_edge_pred = local_model.predict_proba(edge_cases)
    docker_edge_pred = docker_model.predict_proba(edge_cases)
    
    edge_close = np.allclose(local_edge_pred, docker_edge_pred, rtol=tolerance)
    print(f"  Edge case predictions match: {'✅' if edge_close else '❌'}")
    
    # Overall validation result
    all_tests_pass = all([
        params_match,
        predictions_match,
        proba_close,
        accuracy_close,
        auc_close,
        edge_close
    ])
    
    print(f"\n🎉 Overall validation: {'✅ PASS' if all_tests_pass else '❌ FAIL'}")
    
    return all_tests_pass
    

def create_edge_cases(feature_names):
    """Create edge cases for testing model robustness."""
    # Create edge cases based on diabetes dataset characteristics
    edge_cases = pd.DataFrame({
        'Pregnancies': [0, 15, 20],  # Min, high, extreme
        'PlasmaGlucose': [50, 200, 300],  # Low, high, extreme
        'DiastolicBloodPressure': [40, 120, 150],  # Low, high, extreme
        'TricepsThickness': [0, 40, 99],  # Min, high, max
        'SerumInsulin': [0, 300, 800],  # Min, high, extreme
        'BMI': [15, 45, 70],  # Low, high, extreme
        'DiabetesPedigree': [0.1, 1.5, 2.5],  # Low, high, extreme
        'Age': [18, 65, 80]  # Young, older, elderly
    })
    
    return edge_cases

if __name__ == "__main__":
    validate_model_consistency()