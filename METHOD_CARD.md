# Production Customer Churn ML Method Card

## Intended use
Portfolio demonstration of a reproducible training-to-serving ML workflow for binary churn prediction.

## Training and selection
The OpenML Telco Customer Churn benchmark is stratified into 60% train, 20% validation, and 20% locked test. Candidate models are compared on validation ROC-AUC, with PR-AUC and recall as tie-breakers. The operating threshold is selected only on validation and maximizes F1 subject to at least 70% churn recall.

## Model registry metadata
The serialized bundle stores the fitted pipeline, locked threshold, model family, UTC model version, and expected feature columns. A separate JSON metadata record stores training timestamp, dataset audit/fingerprint, selection policy, threshold policy, and locked-test metrics.

## Serving contract
FastAPI exposes health, readiness, and prediction endpoints. Online inference rejects missing and unknown features instead of silently reshaping malformed payloads. Batch scoring preserves the selected model version in output rows.

## Monitoring
Training features and training prediction scores are exported as a reference distribution. Numeric drift uses KS statistic and population stability index (PSI). Categorical drift uses total variation distance. Alert thresholds are heuristic diagnostics rather than universal production limits.

## Limitations
- small historical benchmark rather than company production data;
- random stratified split rather than temporal validation;
- no external feature store, managed registry, orchestrator, or cloud deployment;
- no delayed-label monitoring loop;
- no causal claim that churn scores improve retention;
- no universal claim for drift thresholds;
- Docker/CI demonstrate packaging and automation patterns, not a certified production environment.

## Production extensions
A real deployment would add out-of-time validation, secret/config management, authenticated endpoints, structured logging, durable prediction storage, delayed-label joins, model/performance drift alerts, experiment tracking, managed registry, rollback/canary deployment, SLOs, load tests, orchestration, security scanning, and organization-specific governance.
