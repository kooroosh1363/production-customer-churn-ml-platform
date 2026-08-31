from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"


def score_csv(input_csv: str, output_csv: str):
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_csv)
    expected = bundle["feature_columns"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    prob = bundle["model"].predict_proba(df[expected])[:, 1]
    out = df.copy()
    out["churn_probability"] = prob
    out["churn_prediction"] = (prob >= float(bundle["threshold"])).astype(int)
    out["model_version"] = bundle["model_version"]
    out.to_csv(output_csv, index=False)
    return output_csv
