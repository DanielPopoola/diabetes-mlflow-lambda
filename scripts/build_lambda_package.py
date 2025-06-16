import os
import shutil
import zipfile
import joblib
from pathlib import Path
import mlflow.sklearn
from src.config import MODEL_NAME, LAMBDA_PACKAGE_SIZE_LIMIT, PROJECT_ROOT

def extract_model_from_mlflow(model_name, stage="Production", output_dir="lambda_package"):
    """Extract model from MLflow registry for Lambda deployment."""
    
    # Set MLflow tracking URI for Docker environment
    mlflow.set_tracking_uri("file:///app/mlflow/docker/mlflow.db")
    
    try:
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Create output directory
        output_path = PROJECT_ROOT / output_dir
        output_path.mkdir(exist_ok=True)
        
        # Save model as joblib
        model_path = output_path / "model.joblib"
        joblib.dump(model, model_path)
        
        
        # Copy Lambda handler
        handler_source = PROJECT_ROOT / "src" / "deployment" / "lambda_handler.py"
        handler_dest = output_path / "lambda_function.py"  # AWS Lambda expects this name
        shutil.copy2(handler_source, handler_dest)
        
        # Get model size
        model_size = model_path.stat().st_size
        
        print(f"✅ Model extracted successfully:")
        print(f"  Model size: {model_size / (1024*1024):.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error extracting model: {e}")
        return None

def create_deployment_package():
    """Create Lambda deployment package."""
    
    package_dir = PROJECT_ROOT / "lambda_package"
    
    # Extract model from MLflow
    if not extract_model_from_mlflow(MODEL_NAME):
        return False
    
    # Install dependencies in package directory
    os.system(f"pip install -t {package_dir} --no-deps scikit-learn==1.3.2 numpy==1.24.3 joblib==1.3.2")
    
    # Create zip file
    zip_path = package_dir / "deployment.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            # Skip the zip file itself and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('__pycache__')]
            
            for file in files:
                if file.endswith('.zip'):
                    continue
                    
                file_path = Path(root) / file
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)
    
    # Check size
    zip_size = zip_path.stat().st_size
    
    print(f"\n📦 Deployment package created:")
    print(f"  Package size: {zip_size / (1024*1024):.2f} MB")
    print(f"  Size limit: {LAMBDA_PACKAGE_SIZE_LIMIT / (1024*1024):.0f} MB")
    
    if zip_size > LAMBDA_PACKAGE_SIZE_LIMIT:
        print(f"⚠️ Package exceeds Lambda size limit!")
        return False
    else:
        print(f"✅ Package size OK")
        return True

if __name__ == "__main__":
    create_deployment_package()