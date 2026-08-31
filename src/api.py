from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
app = FastAPI(title="Customer Churn ML API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: dict[str, Any]


def load_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model artifact not found. Run python -m src.train first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"ready": MODEL_PATH.exists()}


@app.post("/predict")
def predict(req: PredictionRequest):
    bundle = load_bundle()
    expected = bundle["feature_columns"]
    missing = [c for c in expected if c not in req.features]
    unknown = [c for c in req.features if c not in expected]
    if missing or unknown:
        raise HTTPException(status_code=422, detail={"missing_features": missing, "unknown_features": unknown})
    row = pd.DataFrame([{c: req.features[c] for c in expected}])
    prob = float(bundle["model"].predict_proba(row)[:, 1][0])
    pred = int(prob >= float(bundle["threshold"]))
    return {
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "churn_probability": prob,
        "threshold": float(bundle["threshold"]),
        "prediction": pred,
    }
