import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def lambda_handler(event, context):
    """AWS Lambda handler for diabetes prediction."""

    if model is None:
        return {
            'statusCode': 500, 
            'body': json.dumps({'error': 'Model not loaded properly'})
        }
    
    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        expected_features = [
            'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
            'TricepsThickness', 'SerumInsulin', 'BMI', 
            'DiabetesPedigree', 'Age'
        ]
    
        if not all(feature in body for feature in expected_features): # type: ignore
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required features',
                    'required': expected_features
                })
            }

        input_data = np.array([[body[feature] for feature in expected_features]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        response = {
            'prediction': int(prediction),
            'probability': {
                'non_diabetic': float(probability[0]),
                'diabetic': float(probability[1])
            },
            'confidence': float(max(probability)),
            'risk_level': 'high' if probability[1] > 0.7 else 'medium' if probability[1] > 0.3 else 'low'
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    
# Local testing function
def test_locally():
    """Test the handler locally."""
    sample_input = {
        'Pregnancies': 6,
        'PlasmaGlucose': 148,
        'DiastolicBloodPressure': 72,
        'TricepsThickness': 35,
        'SerumInsulin': 0,
        'BMI': 33.6,
        'DiabetesPedigree': 0.627,
        'Age': 50
    }
    
    result = lambda_handler(sample_input, {})
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_locally()