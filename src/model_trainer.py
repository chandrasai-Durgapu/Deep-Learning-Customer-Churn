import mlflow
import mlflow.keras

import os
import pandas as pd
import logging
import json
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
from model_builder import build_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_data(train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
    logger.info("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('Churn', axis=1).values
    y_train = train_df['Churn'].values
    
    X_test = test_df.drop('Churn', axis=1).values
    y_test = test_df['Churn'].values
    
    logger.info(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, epochs=30, batch_size=16, patience=5):
    logger.info("Starting training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2
    )
    logger.info("Training completed.")
    return history

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    logger.info(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

def save_model(model, save_dir="saved_models", filename="customer_churn_model.h5"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

def save_metrics(history, filename="metrics.json"):
    metrics = {
        "accuracy": history.history['accuracy'][-1],
        "val_accuracy": history.history['val_accuracy'][-1],
        "loss": history.history['loss'][-1],
        "val_loss": history.history['val_loss'][-1]
    }
    with open(filename, 'w') as f:
        json.dump(metrics, f)
    logger.info(f"Metrics saved to {filename}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = build_model(X_train.shape[1])

    mlflow.set_experiment("customer-churn-prediction")
    mlflow.keras.autolog()

    with mlflow.start_run():
        history = train_model(model, X_train, y_train)
        save_metrics(history)

        evaluate_model(model, X_test, y_test)
        save_model(model)

        mlflow.log_param("epochs", 30)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("patience", 5)
