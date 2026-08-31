# DS-10 — Production Customer Churn ML Platform

Portfolio-grade production ML platform for customer churn with reproducible training, versioned model artifacts, API inference, batch scoring, validation, drift monitoring, automated tests, Docker, and CI/CD-oriented workflows.

## What this project demonstrates

- OpenML Telco Customer Churn benchmark (data id `42178`)
- schema, row-count, target, ID uniqueness, and dataset fingerprint checks
- stratified 60/20/20 train/validation/locked-test design
- Logistic Regression, Random Forest, and HistGradientBoosting candidates
- validation-only model selection and operating-threshold tuning
- versioned `model.joblib` bundle with feature contract and metadata
- FastAPI `/health`, `/ready`, and `/predict` endpoints
- strict missing/unknown feature rejection at online inference
- batch CSV scoring with model-version lineage
- reference-distribution export for monitoring
- numeric KS + PSI drift diagnostics
- categorical total-variation drift diagnostics
- heuristic drift alert summary with explicit limitations
- Dockerized serving
- end-to-end pytest coverage and GitHub Actions CI

## Architecture

```text
OpenML Telco churn
   -> integrity + schema audit
   -> train / validation / locked test
   -> preprocessing
   -> candidate models
   -> validation model selection
   -> validation-only threshold
   -> locked test evaluation
   -> versioned model bundle + metadata
      -> FastAPI online inference
      -> batch scoring
      -> prediction/version lineage
   -> reference distribution
      -> drift diagnostics
   -> tests + CI + Docker
```

## Production boundary

This repository demonstrates production-ML engineering patterns around an offline benchmark. It does not claim that the model is ready for deployment at a real telecom operator, that drift thresholds are universal, or that the model improves retention or revenue without organization-specific validation, monitoring, governance, and controlled rollout.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t churn-ml-platform .
docker run -p 8000:8000 churn-ml-platform
```

See `DATA_SOURCE.md` and `METHOD_CARD.md` for provenance, methodology, monitoring assumptions, and limitations.
