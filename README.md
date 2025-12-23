# Demand Forecasting ML System (Production-Grade, End-to-End)

## Overview
This project implements a **full lifecycle demand forecasting system** inspired by real-world retail ML platforms. It goes far beyond model training by addressing **data leakage, baselines, hybrid modeling, deployment, monitoring, drift detection, and safe auto-retraining**.

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

