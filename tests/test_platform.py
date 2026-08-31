from pathlib import Path
import json
import pandas as pd
from fastapi.testclient import TestClient

from src.data import load_dataset
from src.train import main
from src.api import app
from src.monitoring import monitor, summarize_drift

ROOT = Path(__file__).resolve().parents[1]


def test_training_api_and_monitoring_end_to_end():
    main()
    Xtr, ytr, Xv, yv, Xte, yte, audit = load_dataset(42)
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())

    assert audit["rows"] == 7043
    assert audit["features"] == 19
    assert audit["train_rows"] == 4225
    assert audit["validation_rows"] == 1409
    assert audit["test_rows"] == 1409
    assert metadata["selected_model"] if "selected_model" in metadata else metadata["model_name"]
    assert metadata["threshold_policy"]["recall"] >= 0.70
    assert metadata["test_result"]["roc_auc"] > 0.75
    assert (ROOT / "artifacts" / "model.joblib").exists()
    assert (ROOT / "artifacts" / "reference_distribution.csv").exists()

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["ready"] is True
    payload = {"features": Xte.iloc[0].to_dict()}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["prediction"] in [0, 1]
    assert body["model_version"]

    reference = Xtr.copy()
    current = Xte.copy()
    report = monitor(reference, current)
    summary = summarize_drift(report)
    assert not report.empty
    assert "drift_alert" in summary
