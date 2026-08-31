from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def population_stability_index(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def monitor(reference: pd.DataFrame, current: pd.DataFrame):
    rows = []
    shared = [c for c in reference.columns if c in current.columns]
    for col in shared:
        if pd.api.types.is_numeric_dtype(reference[col]):
            ref = reference[col].dropna().astype(float)
            cur = current[col].dropna().astype(float)
            if len(ref) and len(cur):
                stat, p = ks_2samp(ref, cur)
                rows.append({"feature": col, "type": "numeric", "ks_stat": float(stat), "ks_pvalue": float(p), "psi": population_stability_index(ref, cur)})
        else:
            ref_dist = reference[col].astype(str).value_counts(normalize=True)
            cur_dist = current[col].astype(str).value_counts(normalize=True)
            cats = sorted(set(ref_dist.index) | set(cur_dist.index))
            tvd = 0.5 * sum(abs(float(ref_dist.get(c, 0)) - float(cur_dist.get(c, 0))) for c in cats)
            rows.append({"feature": col, "type": "categorical", "total_variation_distance": float(tvd)})
    return pd.DataFrame(rows)


def summarize_drift(report: pd.DataFrame):
    alerts = []
    for _, row in report.iterrows():
        if row["type"] == "numeric" and (row.get("psi", 0) >= 0.20 or row.get("ks_stat", 0) >= 0.20):
            alerts.append(str(row["feature"]))
        if row["type"] == "categorical" and row.get("total_variation_distance", 0) >= 0.15:
            alerts.append(str(row["feature"]))
    return {"drift_alert": bool(alerts), "alert_features": alerts, "note": "Heuristic diagnostic thresholds; not statistical proof of production degradation."}


def save_report(reference_csv: str, current_csv: str):
    reference = pd.read_csv(reference_csv)
    current = pd.read_csv(current_csv)
    report = monitor(reference, current)
    report.to_csv(ART / "drift_report.csv", index=False)
    summary = summarize_drift(report)
    (ART / "drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
