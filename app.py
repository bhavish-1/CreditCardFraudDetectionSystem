import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI

from src.features.feature_engineering import prepare_features


# ---------------------------
# App init
# ---------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Inference service with automatic model switching",
    version="1.0"
)

# ---------------------------
# Paths
# ---------------------------
MODEL_DIR = "models"
ACTIVE_MODEL_FILE = os.path.join(MODEL_DIR, "active_model.txt")
FEATURE_SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_columns.json")


# ---------------------------
# Load feature schema
# ---------------------------
def load_feature_schema():
    if not os.path.exists(FEATURE_SCHEMA_PATH):
        raise FileNotFoundError("Feature schema not found")

    with open(FEATURE_SCHEMA_PATH, "r") as f:
        return json.load(f)


# ---------------------------
# Load active model
# ---------------------------
def load_active_model():
    if not os.path.exists(ACTIVE_MODEL_FILE):
        raise FileNotFoundError("Active model file not found")

    with open(ACTIVE_MODEL_FILE, "r") as f:
        model_filename = f.read().strip()

    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "Fraud detection service is live"
    }


# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict")
def predict(transaction: dict):
    """
    Predict whether a transaction is fraudulent.
    """

    # 1. Convert input to DataFrame
    input_df = pd.DataFrame([transaction])

    # 2. Feature engineering (raw → features)
    processed_df = prepare_features(input_df, training=False)

    # 3. Load feature schema (from training)
    feature_columns = load_feature_schema()

    # 4. Add missing columns
    for col in feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    # 5. Remove extra columns & enforce order
    processed_df = processed_df[feature_columns]

    # 6. Load active model
    model = load_active_model()

    # 7. Predict probability
    fraud_prob = model.predict_proba(processed_df)[:, 1][0]

    return {
        "fraud_probability": round(float(fraud_prob), 4),
        "is_fraud": int(fraud_prob >= 0.5)
    }
