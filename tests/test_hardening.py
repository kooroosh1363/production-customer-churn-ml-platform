from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.batch import score_csv
from src.data import load_dataset
from src.monitoring import monitor, summarize_drift
from src.train import main

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def test_production_hardening(tmp_path):
    main()
    bundle = joblib.load(ART / "model.joblib")
    metadata = json.loads((ART / "model_metadata.json").read_text())
    Xtr, ytr, Xv, yv, Xte, yte, audit = load_dataset(42)

    # Artifact + metadata consistency.
    assert bundle["model_name"] == metadata["model_name"]
    assert bundle["model_version"] == metadata["model_version"]
    assert abs(float(bundle["threshold"]) - float(metadata["threshold_policy"]["threshold"])) < 1e-12
    assert bundle["feature_columns"] == Xtr.columns.tolist()
    assert len(bundle["feature_columns"]) == audit["features"] == 19
    assert metadata["test_result"]["roc_auc"] > 0.80
    assert metadata["test_result"]["recall"] > 0.70

    # Online API contract and prediction contract.
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is True
    record = Xte.iloc[0].to_dict()
    response = client.post("/predict", json={"features": record})
    assert response.status_code == 200
    payload = response.json()
    direct_prob = float(bundle["model"].predict_proba(Xte.iloc[[0]])[:, 1][0])
    assert abs(payload["churn_probability"] - direct_prob) < 1e-12
    assert payload["model_version"] == bundle["model_version"]
    assert payload["prediction"] == int(direct_prob >= float(bundle["threshold"]))

    missing_record = dict(record)
    missing_record.pop(bundle["feature_columns"][0])
    assert client.post("/predict", json={"features": missing_record}).status_code == 422
    unknown_record = dict(record)
    unknown_record["unexpected_feature"] = 1
    assert client.post("/predict", json={"features": unknown_record}).status_code == 422

    # Batch/online parity on identical records.
    batch_in = tmp_path / "batch_in.csv"
    batch_out = tmp_path / "batch_out.csv"
    Xte.iloc[:5].to_csv(batch_in, index=False)
    score_csv(str(batch_in), str(batch_out))
    scored = pd.read_csv(batch_out)
    expected_prob = bundle["model"].predict_proba(Xte.iloc[:5])[:, 1]
    assert np.allclose(scored["churn_probability"].to_numpy(), expected_prob, atol=1e-12)
    assert scored["model_version"].nunique() == 1
    assert scored["model_version"].iloc[0] == bundle["model_version"]

    # Monitoring should stay quiet on identical data and alert on strong synthetic drift.
    same_report = monitor(Xtr.iloc[:500], Xtr.iloc[:500].copy())
    assert summarize_drift(same_report)["drift_alert"] is False

    shifted = Xtr.iloc[:500].copy()
    numeric_cols = shifted.select_dtypes(include=["number"]).columns.tolist()
    assert numeric_cols
    shifted[numeric_cols[0]] = shifted[numeric_cols[0]].astype(float) + 1000.0
    drift_report = monitor(Xtr.iloc[:500], shifted)
    drift_summary = summarize_drift(drift_report)
    assert drift_summary["drift_alert"] is True
    assert numeric_cols[0] in drift_summary["alert_features"]

    # Claim boundary must remain explicit.
    claim = metadata["claim_boundary"].lower()
    assert "no guarantee" in claim
    assert "organization-specific validation" in claim
