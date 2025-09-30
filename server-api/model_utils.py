import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

# Define paths relative to this file or adjust accordingly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../saved_models/customer_churn_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler.pkl')
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, '../models/label_encoders.pkl')

# Load the scaler, label encoders, and model once on import
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(LABEL_ENCODERS_PATH)
model = load_model(MODEL_PATH)

def preprocess_input(data: dict):
    """
    Preprocess incoming data dict before prediction:
    - Encode categorical features using label encoders
    - Scale features using the saved scaler
    - Return a numpy array ready for prediction
    """
    # Convert input dict to DataFrame (single row)
    df = pd.DataFrame([data])
    
    # Encode categorical columns using saved label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels by assigning -1 or similar
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # Scale the features
    X_scaled = scaler.transform(df)
    
    return X_scaled

def predict_churn(data: dict):
    """
    Accept raw input data dict, preprocess and predict churn probability
    """
    processed_data = preprocess_input(data)
    prob = model.predict(processed_data)[0][0]
    prediction = int(prob > 0.5)
    
    return {
        'churn_probability': float(prob),
        'churn_prediction': prediction
    }
