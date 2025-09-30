import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import os

def load_params(path="params.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_and_split():
    params = load_params()
    raw_path = params["data"]["raw_path"]

    df = pd.read_csv(raw_path)
    df.drop_duplicates(inplace=True)

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert and clean
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrame for saving
    df_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    df_train['Churn'] = y_train.reset_index(drop=True)

    df_test = pd.DataFrame(X_test_scaled, columns=X.columns)
    df_test['Churn'] = y_test.reset_index(drop=True)

    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed", exist_ok=True)

    # Save scaler and label encoders
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")

    # Save processed data to CSV
    df_train.to_csv("data/processed/train.csv", index=False)
    df_test.to_csv("data/processed/test.csv", index=False)

    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    preprocess_and_split()
