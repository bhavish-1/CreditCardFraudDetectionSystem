# 💳 Credit Card Fraud Detection with Automated Drift Handling & Model Deployment

## 📌 Overview
This project implements a production-ready supervised machine learning pipeline for detecting fraudulent credit card transactions from highly imbalanced data. The system continuously monitors incoming data for data drift, triggers automated model retraining, selects the best-performing model, and deploys it to production without manual intervention.

---

## 🚀 Key Features
- End-to-end automated ML lifecycle
- Data drift detection on incoming transaction data
- Automated retraining and evaluation of multiple models
- Dynamic model selection and deployment
- Designed for real-world, production-scale fraud detection

---

## 🏗️ System Workflow
1. Deploy an initial fraud detection model to production  
2. Monitor incoming transaction data  
3. Detect data drift using statistical thresholds  
4. Trigger retraining when drift is detected  
5. Train and evaluate multiple models:
   - Logistic Regression
   - Random Forest
   - XGBoost
6. Automatically select the best-performing model
7. Deploy the selected model to production

---

## 🧠 Models Used
- Logistic Regression (baseline and interpretable)
- Random Forest (robust ensemble model)
- XGBoost (high-performance gradient boosting)

---

## ⚖️ Handling Class Imbalance
- Feature engineering on transaction data
- Class weighting / resampling strategies
- Imbalance-aware evaluation metrics

---

## 📊 Model Evaluation Metrics
- Precision
- Recall
- F1-Score
- ROC-AUC

Accuracy is avoided due to extreme class imbalance.

---

## 🔁 Data Drift Detection
- Incoming data distributions are compared with training data
- Drift is flagged when feature deviations exceed a defined threshold
- Retraining is triggered only when necessary to avoid unnecessary computation

---

## ⚙️ Automation & Deployment
- Automated retraining and evaluation pipeline
- Best-performing model is dynamically selected
- Model deployment occurs without manual intervention
- Production-oriented system design

---

## 🧩 Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- XGBoost

---

## 📈 Outcome
- Improved fraud detection reliability under evolving data patterns
- Reduced performance degradation due to data drift
- Scalable and maintainable ML pipeline aligned with industry best practices

---

## 🔮 Future Enhancements
- Concept drift detection
- Model explainability (SHAP)
- Real-time streaming support
- Monitoring dashboards
