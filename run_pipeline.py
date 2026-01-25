import os
import json
import joblib
import pandas as pd
from datetime import datetime

from src.eda.drift_detection import detect_drift
from src.training.train_models import train_all_models
from src.training.model_selection import select_best_model


DATA_DIR = "/Users/bhavish/Desktop/CreditCardFraudDetectinSystem/data"
MODEL_DIR = "/Users/bhavish/Desktop/CreditCardFraudDetectinSystem/models"

HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, "raw", "fraudTrain.csv")
NEW_DATA_PATH = os.path.join(DATA_DIR, "raw", "temp_1.csv")

ACTIVE_MODEL_FILE = os.path.join(MODEL_DIR, "active_model.txt")

RETRAIN_SAMPLE_THRESHOLD = 10_000
DRIFT_NUMERIC_COLS = ["amt", "city_pop"]


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def save_model(model, metrics, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"{model_name}_{timestamp}.pkl"
    metrics_filename = f"{model_name}_{timestamp}_metrics.json"

    model_path = os.path.join(MODEL_DIR, model_filename)
    metrics_path = os.path.join(MODEL_DIR, metrics_filename)

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return model_filename


def update_active_model(model_filename):
    with open(ACTIVE_MODEL_FILE, "w") as f:
        f.write(model_filename)


def run_pipeline():
    print("🚀 Starting fraud detection pipeline")

    historical_df = load_csv(HISTORICAL_DATA_PATH)
    new_df = load_csv(NEW_DATA_PATH)

    print(f"Historical samples: {len(historical_df)}")
    print(f"New batch samples: {len(new_df)}")

    if len(new_df) < RETRAIN_SAMPLE_THRESHOLD:
        print("❌ Not enough new data. Skipping retraining.")
        return

    drift_report, drift_detected = detect_drift(
        reference_df=historical_df,
        new_df=new_df,
        numeric_cols=DRIFT_NUMERIC_COLS
    )

    print("Drift report:")
    print(json.dumps(drift_report, indent=2))

    if not drift_detected:
        print("✅ No significant drift detected. Skipping retraining.")
        return

    print("⚠️ Drift detected. Retraining models...")

    combined_df = pd.concat([historical_df, new_df], ignore_index=True)

    models = train_all_models(combined_df)

    best_name, best_model, best_metrics = select_best_model(models)

    print(f"🏆 Best model selected: {best_name}")
    print(f"Metrics: {best_metrics}")

    model_filename = save_model(best_model, best_metrics, best_name)
    update_active_model(model_filename)

    print(f"✅ Model promoted: {model_filename}")


if __name__ == "__main__":
    run_pipeline()
