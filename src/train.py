from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load_dataset

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RANDOM_STATE = 42


def preprocessor(X: pd.DataFrame):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ], verbose_feature_names_out=False)


def candidates(X: pd.DataFrame):
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor(X)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor(X)),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("prep", preprocessor(X)),
            ("model", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, max_leaf_nodes=15, random_state=RANDOM_STATE)),
        ]),
    }


def eval_metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def choose_threshold(y, prob, min_recall=0.70):
    best = None
    for t in np.unique(prob):
        m = eval_metrics(y, prob, t)
        if m["recall"] >= min_recall:
            key = (-m["f1"], -m["precision"], float(t))
            if best is None or key < best[0]:
                best = (key, float(t), m)
    if best is None:
        return 0.5, eval_metrics(y, prob, 0.5)
    return best[1], best[2]


def main():
    ART.mkdir(exist_ok=True)
    Xtr, ytr, Xv, yv, Xte, yte, audit = load_dataset(RANDOM_STATE)
    fitted, rows = {}, []
    for name, model in candidates(Xtr).items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        row = eval_metrics(yv, model.predict_proba(Xv)[:, 1])
        row["model"] = name
        rows.append(row)

    val = pd.DataFrame(rows).sort_values(["roc_auc", "pr_auc", "recall"], ascending=False).reset_index(drop=True)
    selected = str(val.iloc[0]["model"])
    model = fitted[selected]
    val_prob = model.predict_proba(Xv)[:, 1]
    threshold, threshold_metrics = choose_threshold(yv, val_prob)
    test_prob = model.predict_proba(Xte)[:, 1]
    test_metrics = eval_metrics(yte, test_prob, threshold)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = {
        "model": model,
        "threshold": threshold,
        "model_name": selected,
        "model_version": version,
        "feature_columns": Xtr.columns.tolist(),
    }
    joblib.dump(bundle, ART / "model.joblib")
    val.to_csv(ART / "validation_metrics.csv", index=False)
    reference = Xtr.copy()
    reference["prediction"] = model.predict_proba(Xtr)[:, 1]
    reference.to_csv(ART / "reference_distribution.csv", index=False)

    metadata = {
        "model_name": selected,
        "model_version": version,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_audit": audit,
        "selection_policy": "highest validation ROC-AUC; PR-AUC then recall tie-break",
        "threshold_policy": {"minimum_validation_recall": 0.70, "threshold": threshold, **threshold_metrics},
        "test_result": test_metrics,
        "claim_boundary": "offline benchmark plus production-engineering demonstration; no guarantee of live retention lift, revenue impact, or deployment fitness without organization-specific validation",
    }
    (ART / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
