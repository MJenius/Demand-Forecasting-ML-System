# Demand Forecasting ML System (Production-Grade, End-to-End)

## Overview
This project implements a **full lifecycle demand forecasting system** inspired by real-world retail ML platforms. It goes far beyond model training by addressing **data leakage, baselines, hybrid modeling, deployment, monitoring, drift detection, and safe auto-retraining**.

The production entry point is `production_pipeline.py`. It validates the raw M5 inputs, builds horizon-safe features, runs expanding-window validation at 7- and 28-day horizons, compares seasonal-naive, 28-day moving average, Ridge, and LightGBM forecasts, measures PSI drift, applies guarded retraining, records runtime/memory, and logs the complete run to local MLflow storage. Existing inference and registry components remain available.

## Reproducible run

Python 3.12 is required. From a clean checkout:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python production_pipeline.py
.venv\Scripts\python -m pytest -q
```

The tracked M5 archive is extracted automatically if the raw CSVs are absent. `pipeline.toml` is the full 3,049-item configuration and uses chunked float32 Parquet features plus bounded LightGBM training batches. `pipeline.dev.toml` runs the deterministic development benchmark: **100 sampled items across all 10 stores and the full available historical period**. Outputs are written to `artifacts/`, experiments to `mlruns/`, and both are reproducible generated files rather than source-controlled results. A failed full run records `artifacts/checkpoint.json`; after correcting the cause, `production_pipeline.py --resume` reuses validated content-addressed source and feature caches, then safely reruns evaluation.

Data quality fails closed on missing or changed columns, null/negative demand, duplicate calendar or price keys, invalid dates, non-positive prices, and stale sales/calendar alignment. PSI thresholds, validation folds, horizons, model parameters, and promotion tolerance are configurable. Promotion requires the candidate to improve MAE without degrading RMSE or MAPE on the same holdout.

## Measured full-data benchmark

The 2026-08-16 production run used all **3,049 items**, **10 stores**, **30,490 item-store series**, and **58,327,370 historical records** from 2011-01-29 through 2016-04-24. Metrics are three-fold expanding-window means; MAPE excludes zero actuals.

| Horizon | Model | MAE | RMSE | MAPE |
|---:|---|---:|---:|---:|
| 7 days | LightGBM | 1.0384 | 2.4940 | 65.8768% |
| 7 days | Moving average | 1.0973 | 2.3975 | 57.6849% |
| 7 days | Ridge | 1.1384 | 2.3450 | 54.6909% |
| 7 days | Seasonal naive | 1.3540 | 3.0130 | 84.4741% |
| 28 days | LightGBM | 1.1073 | 2.7029 | 66.4901% |
| 28 days | Moving average | 1.1546 | 2.5549 | 60.0618% |
| 28 days | Ridge | 1.2172 | 2.5081 | 53.3869% |
| 28 days | Seasonal naive | 1.4189 | 3.1994 | 87.2804% |

LightGBM achieved the lowest MAE, improving on the best MAE baseline (moving average) by **5.36% at 7 days** and **4.10% at 28 days**; Ridge achieved the lowest RMSE and MAPE. The successful cache-recovery run took **2,236.39 seconds**, peaked at **786.89 MB working set**, and evaluated 182,940 forecast cases at **81.80 cases/second**. A cold full-data build separately peaked at **1,805.90 MB**. Drift triggered retraining, and the candidate was rejected because RMSE degraded by 0.585%.

The system is designed to answer a realistic business question:

> **“How many units of each product should be stocked at each store in the near future, and how can we ensure the model stays reliable as demand patterns change?”**

This repository demonstrates how modern ML systems are built, operated, and protected in production.

---

## Business Problem
Retailers face two costly risks:
- **Under-forecasting** → stock-outs, lost sales
- **Over-forecasting** → excess inventory, holding costs

Demand forecasting is difficult because:
- Demand is noisy and sparse (many low-volume SKUs)
- Patterns change over time (seasonality, promotions, pricing)
- A single global ML model often performs poorly across all SKUs

This project addresses these challenges using **baselines, segmentation, monitoring, and automated decision logic**, not just a single ML model.

---

## Key Ideas & Uniqueness

### 1. Baselines First (Not Optional)
Before any ML model is trained, the system establishes **strong statistical baselines**:
- Last-value forecast
- 7-day moving average
- Seasonal naive (weekly)

These baselines set a **minimum performance bar**. Any ML model must beat them to be considered useful.

---

### 2. Leakage-Safe Feature Engineering
All features are:
- Time-shifted
- Rolling-window based
- Validated explicitly against baseline reproduction

A dedicated validation step reproduces baseline metrics *exactly* from the feature set, proving **zero data leakage**.

---

### 3. Hybrid Two-Model Strategy (Core Innovation)
A single model performs poorly across all SKUs due to demand heterogeneity.

This system uses **SKU segmentation**:
- **Low-volume SKUs (≈82%)** → statistical 7-day moving average
- **Normal-volume SKUs (≈18%)** → LightGBM regression model

Routing logic selects the appropriate model **at inference time**.

This hybrid approach achieved:
- **+25% MAE improvement**
- **+7% RMSE improvement**

---

### 4. Model Registry & Versioning
Models are never overwritten.

Each version stores:
- Metrics
- Training window
- Feature list
- SKU segmentation thresholds

Only one model version is active at a time, enabling:
- Rollbacks
- Auditing
- Safe upgrades

---

### 5. Production-Style Inference API
A FastAPI service provides:
- `/health` endpoint
- `/model-info` endpoint
- `/predict` endpoint

Predictions include:
- Model used (LightGBM or MA)
- Model version

Every request is logged for monitoring.

---

### 6. Monitoring & Drift Detection
The system continuously observes:
- **Prediction distributions**
- Mean, variance, percentiles

Drift is detected **without labels** using distribution shifts.

Drift states:
- OK
- WARNING
- DRIFT

---

### 7. Safe Auto-Retraining (Closed Loop)
When drift is detected:
1. Retraining is triggered
2. A candidate model is trained
3. Candidate is evaluated vs current model
4. Promotion occurs **only if metrics improve**

In testing, a degraded candidate model was correctly **rejected**, proving system stability.

---

## System Architecture

```
                 ┌─────────────────────────┐
                 │     Raw Retail Data      │
                 │   (Sales, Calendar,      │
                 │    Prices – M5 Dataset)  │
                 └───────────┬─────────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │  Data Ingestion &        │
                 │  Reshaping (Long Format) │
                 └───────────┬─────────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │ Leakage-Safe Feature     │
                 │ Engineering Pipeline     │
                 │ (Lags, Rolling, Calendar)│
                 └───────────┬─────────────┘
                             │
                             ▼
        ┌───────────────────────────────┐
        │  Baselines (7-Day MA, Seasonal)│
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ LightGBM Model (Normal SKUs)   │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ Hybrid Routing Logic           │
        │ (SKU Segmentation)             │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ Model Registry (Versioned)     │
        │ v1, v2, ...                    │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ FastAPI Inference Service      │
        │ /predict, /health              │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ Prediction Logs (JSONL/DB)     │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ Drift Detection                │
        │ (Prediction Distribution)      │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │ Auto-Retraining Pipeline       │
        │ (Train → Evaluate → Promote)   │
        └───────────────────────────────┘
```

---

## Technology Stack

- **Language:** Python
- **Modeling:** LightGBM
- **API:** FastAPI
- **Data:** Pandas, Parquet
- **Storage:** SQLite / JSONL
- **Monitoring:** Custom drift detection
- **Versioning:** Custom model registry

---

## Why This Project Is Different

Most ML projects:
- Stop at training a model
- Ignore baselines
- Ignore drift
- Ignore deployment safety

This project:
- Treats ML as a **system**, not a notebook
- Explicitly handles failure cases
- Prevents bad models from deploying
- Mirrors real-world ML engineering practices

---

## Key Takeaway

> **Good ML systems are not defined by their best model, but by how safely they handle change.**

This repository demonstrates exactly that.

---

## Possible Extensions
- Feature drift detection
- Online performance monitoring (once labels arrive)
- CI/CD integration
- Cloud deployment

---

## Author Notes
This project was built to demonstrate **end-to-end ML engineering competence**, not just predictive modeling. Every design decision prioritizes robustness, explainability, and production realism.
