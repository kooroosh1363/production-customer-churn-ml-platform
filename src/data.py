from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "telco_customer_churn.csv"
OPENML_ID = 42178
EXPECTED_ROWS = 7043
RANDOM_STATE = 42
TARGET = "Churn"
ID_COL = "customerID"


def _frame_fingerprint(df: pd.DataFrame) -> str:
    payload = df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_column(columns, expected: str) -> str | None:
    lookup = {str(c).strip().lower(): c for c in columns}
    return lookup.get(expected.lower())


def load_dataset(random_state: int = RANDOM_STATE):
    RAW.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        df = pd.read_csv(CACHE)
    else:
        bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        df.to_csv(CACHE, index=False)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    target_col = _resolve_column(df.columns, TARGET)
    id_col = _resolve_column(df.columns, ID_COL)
    total_charges_col = _resolve_column(df.columns, "TotalCharges")

    if target_col is None:
        raise ValueError(f"Expected target column {TARGET!r}; found columns: {list(df.columns)}")
    if total_charges_col is None:
        raise ValueError("Expected TotalCharges feature")

    # Some OpenML distributions omit the customer identifier because it is not a
    # predictive feature. Treat it as optional rather than failing the pipeline.
    identifier_available = id_col is not None
    if identifier_available and df[id_col].duplicated().any():
        raise ValueError("customerID must be unique when present")

    df[total_charges_col] = pd.to_numeric(df[total_charges_col], errors="coerce")
    missing_total_charges = int(df[total_charges_col].isna().sum())
    df[total_charges_col] = df[total_charges_col].fillna(0.0)

    y = df[target_col].astype(str).str.strip().map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected churn labels")

    drop_cols = [target_col] + ([id_col] if identifier_available else [])
    X = df.drop(columns=drop_cols).copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    audit = {
        "openml_id": OPENML_ID,
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "churn_rows": int(y.sum()),
        "churn_rate": float(y.mean()),
        "identifier_available": bool(identifier_available),
        "missing_total_charges_coerced": missing_total_charges,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "split_policy": "stratified 60/20/20 with locked test set",
        "dataset_fingerprint_sha256": _frame_fingerprint(df),
    }
    return (
        X_train.reset_index(drop=True), y_train.reset_index(drop=True),
        X_val.reset_index(drop=True), y_val.reset_index(drop=True),
        X_test.reset_index(drop=True), y_test.reset_index(drop=True), audit
    )
