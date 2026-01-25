import time
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from xgboost import XGBClassifier

from src.features.feature_engineering import prepare_features


def compute_metrics(model, X_test, y_test):
    """
    Compute ROC-AUC and PR-AUC for a trained model.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    return {
        "roc_auc": round(roc, 6),
        "pr_auc": round(pr_auc, 6)
    }


def train_all_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train Logistic Regression, Random Forest, and XGBoost models.
    """

    processed_df = prepare_features(df, training=True)

    FEATURE_COLUMNS = processed_df.drop(columns=["is_fraud"]).columns.tolist()
    with open("models/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)


    X = processed_df.drop(columns=['is_fraud'])
    y = processed_df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    neg, pos = y_train.value_counts()
    scale_pos_weight = neg / pos

    models = {}

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        solver='saga',
        n_jobs=-1
    )

    start = time.perf_counter()
    lr.fit(X_train, y_train)
    lr_time = time.perf_counter() - start

    models['logistic_regression'] = {
        "model": lr,
        "metrics": compute_metrics(lr, X_test, y_test),
        "train_time_sec": round(lr_time, 2)
    }

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state
    )

    start = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - start

    models['random_forest'] = {
        "model": rf,
        "metrics": compute_metrics(rf, X_test, y_test),
        "train_time_sec": round(rf_time, 2)
    }

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='aucpr',
        n_jobs=-1,
        random_state=random_state
    )

    start = time.perf_counter()
    xgb.fit(X_train, y_train)
    xgb_time = time.perf_counter() - start

    models['xgboost'] = {
        "model": xgb,
        "metrics": compute_metrics(xgb, X_test, y_test),
        "train_time_sec": round(xgb_time, 2)
    }

    return models
