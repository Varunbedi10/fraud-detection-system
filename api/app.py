from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pathlib

app = FastAPI(title="Fraud Detection API")

# Resolve correct absolute paths
ROOT = pathlib.Path(__file__).resolve().parent.parent
model_path = ROOT / "models" / "xgb_best.pkl"
scaler_path = ROOT / "models" / "scaler.pkl"

# Load model + scaler safely
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler: {e}")

# Transaction Schema (30 features)
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/health")
def health():
    return {"status": "alive"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = np.array([[getattr(transaction, field) for field in transaction.__annotations__]])
        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        return {
            "fraud": int(prediction),
            "probability": round(float(probability), 5)
        }

    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Validation failed", "details": e.errors()},
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "details": str(e)},
        )
