from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model and preprocessing tools
model = load_model("saved_models/customer_churn_model.h5", compile=False)
scaler = joblib.load("models/scaler.pkl")

# Optional: Load label encoders if you have categorical features
try:
    label_encoders = joblib.load("models/label_encoders.pkl")
except FileNotFoundError:
    label_encoders = {}

# Define input schema
class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=0)
    balance: float = Field(..., ge=0)
    products_number: int = Field(..., ge=1)
    credit_score: int = Field(..., ge=300, le=850)
    is_active_member: int = Field(..., ge=0, le=1)
    estimated_salary: float = Field(..., ge=0)
    # Example: Add gender if you have it in training
    # gender: str

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Prepare features in correct order
        raw_features = [
            customer.age,
            customer.tenure,
            customer.balance,
            customer.products_number,
            customer.credit_score,
            customer.is_active_member,
            customer.estimated_salary,
        ]

        # Create DataFrame with matching column names
        input_df = pd.DataFrame([raw_features], columns=[
            "age", "tenure", "balance", "products_number", "credit_score", "is_active_member", "estimated_salary"
        ])

        # Scale input
        scaled_features = scaler.transform(input_df)

        # Predict
        churn_prob = model.predict(scaled_features)[0][0]
        churn_prediction = int(churn_prob > 0.5)

        return {
            "churn_probability": round(float(churn_prob), 4),
            "churn_prediction": churn_prediction
        }

    except Exception as e:
        return {"error": str(e)}
