# 💳 Credit Card Fraud Detection System  
**End-to-End Machine Learning Project | Deployed & Production-Ready**

---

## 🔍 Overview
Credit card fraud causes **huge financial losses** for banks and customers.  
This project builds a **real-time fraud detection system** that analyzes transaction behavior and flags suspicious transactions **before money is lost**.

Unlike a notebook-only ML project, this system is **fully deployed**, **cloud-hosted**, and **usable by non-technical users** through a dashboard.

---

## 🎯 Problem Statement
- Banks process millions of transactions daily  
- Fraud transactions are **extremely rare (~0.17%)**
- Missing a fraud = **direct financial loss**

Traditional accuracy-based models fail here.  
This project focuses on **recall, risk detection, and real-world usability**.

---

## 🧠 Solution
The system:
- Predicts whether a transaction is **Fraud or Safe**
- Returns a **fraud probability score**
- Works in **real time**
- Provides a **business-friendly dashboard** for fraud analysts

---

## 🏗️ System Architecture
Transaction Data
↓
FastAPI Inference Service
↓
Scaler + XGBoost Model
↓
Fraud Probability & Decision
↓
Streamlit Dashboard (for humans)


---

## 📊 Dataset Information
- **Source:** European Credit Card Transactions Dataset  
- **Total records:** 284,807  
- **Fraud cases:** 492 (0.17%)  
- Dataset is **fully anonymized**

### Columns
| Column | Description |
|------|------------|
| Time | Seconds since first transaction (not real timestamp) |
| Amount | Transaction amount |
| V1–V28 | PCA-transformed anonymized features |
| Class | 0 = Normal, 1 = Fraud |

⚠️ Because features are anonymized, this model **cannot accept random credit card CSV files** without retraining.

---

## 🤖 Machine Learning Approach

### Models Tried
- Logistic Regression (baseline – failed on imbalance)
- Random Forest
- **XGBoost (final model)**

### Why XGBoost?
- Handles **non-linear fraud patterns**
- Performs well on **highly imbalanced data**
- Fast inference → suitable for real-time systems

### Evaluation Metrics
| Metric | Reason |
|------|-------|
| Recall | Missing fraud = money lost |
| Precision | Avoid false alarms |
| PR-AUC | Best metric for imbalanced datasets |
| ROC-AUC | Overall class separation |

### Final Performance
- **ROC-AUC:** ~0.97  
- **PR-AUC:** ~0.89  
- High fraud recall with acceptable false positives

---

## 🚀 Backend API (FastAPI)

### Endpoints
- `POST /predict` → Fraud prediction + probability
- `GET /health` → Service health check

### Example Response
```json
{
  "fraud": 1,
  "probability": 0.91
}
https://fraud-detection-system-production.up.railway.app
fraud-detection-system/
│
├── api/                # FastAPI backend
│   └── app.py
│
├── dashboard/          # Streamlit UI
│   └── app.py
│
├── models/             # Trained ML artifacts
│   ├── xgb_best.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── Dockerfile
├── requirements.txt
└── README.md
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run FastAPI Backend
cd api
uvicorn app:app --reload --port 8000
