import pandas as pd

def validate_data(df):
    errors = []

    # Convert TotalCharges to numeric and fill missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['TotalCharges'].isnull().any():
        print("Filling missing TotalCharges with mean value.")
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    # Encode Churn column from Yes/No to 1/0 if needed
    if df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Required columns (excluding customerID)
    required_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'numAdminTickets',
        'numTechTickets', 'Churn'
    ]

    # Check for missing columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check for nulls in required columns
    null_counts = df[required_columns].isnull().sum()
    null_cols = null_counts[null_counts > 0].index.tolist()
    if null_cols:
        errors.append(f"Columns with null values: {null_cols}")

    # Check specific column constraints

    # SeniorCitizen must be 0 or 1
    if not df['SeniorCitizen'].dropna().isin([0, 1]).all():
        errors.append("SeniorCitizen must be 0 or 1")

    # Churn must be 0 or 1
    if not df['Churn'].dropna().isin([0, 1]).all():
        errors.append("Churn must be 0 or 1")

    # MonthlyCharges must be non-negative
    if (df['MonthlyCharges'] < 0).any():
        errors.append("MonthlyCharges must be non-negative")

    # TotalCharges must be non-negative
    if (df['TotalCharges'] < 0).any():
        errors.append("TotalCharges must be non-negative")

    if errors:
        print("Data validation failed:")
        for e in errors:
            print(f" - {e}")
        raise ValueError("Data validation errors found.")

    print("Data validation passed.")

if __name__ == "__main__":
    df = pd.read_csv("data/Customer-Churn-Dataset1.csv")
    validate_data(df)
