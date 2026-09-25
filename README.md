# Enterprise Demand Forecasting & ML Operations Platform

[![CI](https://github.com/MJenius/Demand-Forecasting-ML-System/actions/workflows/ci.yml/badge.svg)](https://github.com/MJenius/Demand-Forecasting-ML-System/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements an **Enterprise Demand Forecasting & ML Operations Platform** built on the Walmart M5 retail forecasting dataset. It demonstrates how modern production ML platforms are engineered beyond offline training notebooks by enforcing:

1. **Strict Data Quality Contracts** with fail-closed schema drift and freshness validation.
2. **Leakage-Safe Feature Engineering** verified by mathematical invariance tests across arbitrary forecast horizons.
3. **Multi-Model Backtesting Arena** comparing Statistical Baselines (Seasonal Naive, 28-Day Moving Average), Linear Models (Streaming Ridge), Gradient Boosted Trees (LightGBM), and **Segmented SKU-Routing Hybrids** across multi-step horizons ($h \in \{7, 28\}$ days).
4. **Cryptographic Lineage Manifests** linking raw dataset checksums, git commits, feature configurations, hyperparameters, and binary weights into immutable SHA-256 identifiers.
5. **Deterministic & Configurable Promotion Gates** evaluating candidate models against active incumbents across MAE, RMSE, WAPE, and Forecast Bias guardrails.
6. **Production Serving API (FastAPI)** serving the active registry model with sub-50ms latency, dynamic feature assembly, and structured prediction logging.
7. **Closed-Loop Operational Monitoring** pairing live prediction streams with delayed actual sales in SQLite to continuously calculate real-time WAPE, Bias, and Population Stability Index (PSI) distribution drift.
8. **Guarded Auto-Retraining** that checks whether retraining improves metrics before safely promoting or automatically rolling back to the active model.

---

## Architecture

```
                  ┌──────────────────────────────┐
                  │    Raw Retail Ingestion      │
                  │   (M5 Sales, Calendar, Sell) │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Data Quality & Schema Gate   │
                  │ (Fail-closed on nulls/drift) │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Leakage-Safe Feature Engine  │
                  │ (Lags, Shifted Rollings, M5) │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  Temporal Expanding Backtest │
                  │  Arena (3 Folds, h=7, h=28)  │
                  └──────────────┬───────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
    [Seasonal Naive]     [Moving Average]       [Streaming Ridge]
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            [Global LightGBM]      [Segmented Hybrid]
                    │                         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  Model Registry & Lineage    │
                  │  (SHA-256 Provenance Hash)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Deterministic Promotion Gate │
                  │ (MAE, RMSE, WAPE, Bias)      │
                  └──────────────┬───────────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
               [REJECT: Keep]         [PROMOTE: Deploy]
                                             │
                                             ▼
                  ┌──────────────────────────────────────┐
                  │ FastAPI Production Serving (/predict)│
                  └──────────────────┬───────────────────┘
                                     │
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ Structured Prediction Logger (SQLite)│
                  └──────────────────┬───────────────────┘
                                     │
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ Delayed Actuals Reconciler           │
                  │ (Operational WAPE, Bias & PSI Drift) │
                  └──────────────────┬───────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
              [State: OK / WARN]             [State: DRIFT]
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │ Closed-Loop Retraining Flow  │
                                      └──────────────────────────────┘
```

---

## Verified Quantitative Benchmarks

The platform enforces transparent reporting between the **100-item standardized 5-model development benchmark** and the **historical 3,049-item full production scale run**. Neither benchmark is conflated with the other, and each model demonstrates distinct operational trade-offs rather than a single universal winner.

### 1. Standardized 100-Item 5-Model Development Benchmark (`pipeline.dev.toml`)
*Setup*: 100 sampled items across all 10 stores (1,000 item-store series), full historical period (2011–2016), 3-fold expanding-window cross-validation, evaluated at $h=7$ and $h=28$ days across all 5 benchmarked model architectures.

| Horizon | Model | MAE | RMSE | WAPE | Forecast Bias | MAPE |
|---:|---|---:|---:|---:|---:|---:|
| **7 days** | **LightGBM** | **0.9770** | 1.9803 | **0.6092** | -0.2701 | 63.35% |
| **7 days** | **Hybrid (Segmented)** | 1.0130 | 1.9557 | 0.6321 | -0.1686 | 54.08% |
| **7 days** | Moving Average (28-day) | 1.0319 | 1.9743 | 0.6433 | -0.1630 | 54.37% |
| **7 days** | **Streaming Ridge** | 1.0852 | **1.9447** | 0.6774 | **-0.0235** | **52.32%** |
| **7 days** | Seasonal Naive | 1.2927 | 2.5276 | 0.8079 | +0.0529 | 84.75% |
| **28 days** | **LightGBM** | **1.0410** | 2.0692 | **0.6474** | -0.2660 | 65.74% |
| **28 days** | **Hybrid (Segmented)** | 1.0771 | 2.0448 | 0.6712 | -0.1577 | 56.75% |
| **28 days** | Moving Average (28-day) | 1.1156 | 2.2097 | 0.6947 | -0.1616 | 57.89% |
| **28 days** | **Streaming Ridge** | 1.1968 | 2.1762 | 0.7463 | **+0.0134** | **52.89%** |
| **28 days** | Seasonal Naive | 1.3660 | 2.6913 | 0.8535 | +0.0595 | 84.14% |

**Empirical Trade-off Analysis**:
- **LightGBM** achieves the lowest point prediction error across both horizons (**MAE 0.9770** at 7d, **1.0410** at 28d; **WAPE 0.6092** and **0.6474**), making it the strongest candidate when minimizing magnitude error. However, it exhibits higher negative forecast bias (-0.2701).
- **Streaming Ridge** achieves the best variance and bias control (**RMSE 1.9447** at 7d; **Forecast Bias -0.0235** at 7d and **+0.0134** at 28d; **MAPE 52.32%** and **52.89%**), avoiding systematic under/over-forecasting.
- **Segmented Hybrid Routing** acts as an intermediary trade-off: it improves upon simple baseline MAE while dampening GBDT bias (-0.1686 vs -0.2701 at 7d), demonstrating how routing high-velocity SKUs to GBDT and intermittent SKUs to linear/statistical baselines balances error and bias.

---

### 2. Historical Full-Data 3,049-Item Scale Benchmark (`pipeline.toml`)
*Setup*: All 3,049 items, 10 stores, 30,490 series, 58,327,370 historical rows, executed using `streaming_pipeline.py` with float32 row-group streaming under 2 GB RAM. *(Note: Evaluates monolithic full-scale models; does not include the 100-item segmented routing).*

| Horizon | Model | MAE | RMSE | MAPE |
|---:|---|---:|---:|---:|
| 7 days | **LightGBM** | **1.0384** | 2.4940 | 65.88% |
| 7 days | Moving Average | 1.0973 | 2.3975 | 57.68% |
| 7 days | Ridge (Streaming) | 1.1384 | 2.3450 | 54.69% |
| 7 days | Seasonal Naive | 1.3540 | 3.0130 | 84.47% |
| 28 days | **LightGBM** | **1.1073** | 2.7029 | 66.49% |
| 28 days | Moving Average | 1.1546 | 2.5549 | 60.06% |
| 28 days | Ridge (Streaming) | 1.2172 | 2.5081 | 53.39% |
| 28 days | Seasonal Naive | 1.4189 | 3.1994 | 87.28% |

*(Full benchmark completed in 2,236s, 81.80 series/s throughput, peak working set 786.89 MB).*

---

## Reproducibility & Quickstart

### Prerequisites
- Python 3.11 or 3.12
- Git

### 1. Setup Environment
```powershell
# Clone repository
git clone https://github.com/MJenius/Demand-Forecasting-ML-System.git
cd "Demand-Forecasting-ML-System"

# Create virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install fastapi uvicorn pydantic pyyaml scipy httpx
```

### 2. Run Automated Test Suite
```powershell
.venv\Scripts\python -m pytest -v
```
*(All 27 unit and integration tests validate data quality, leakage invariants, metrics, registry promotion gates, API serving, reconciler actuals, and lifecycle end-to-end).*

### 3. Run Benchmark Pipeline
```powershell
# Fast development benchmark (100 items across 10 stores, 3-fold backtesting)
.venv\Scripts\python production_pipeline.py --config pipeline.dev.toml
```
Outputs generated in `artifacts/`:
- `artifacts/model_comparison.csv`: Complete multi-model comparison table.
- `artifacts/pipeline_report.json`: Execution manifest with timings and lineage.
- `artifacts/drift_report.json`: PSI distribution analysis across folds.
- `registry/models/v{N}/lineage.json`: Cryptographic lineage manifest.

### 4. Run Production Inference API
```powershell
.venv\Scripts\python -m uvicorn inference.app:app --host 0.0.0.0 --port 8000
```
Test with curl or PowerShell:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
Invoke-RestMethod -Uri "http://localhost:8000/info"
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"item_id":"FOODS_3_090","store_id":"CA_3","date":"2016-04-25"}'
```

---

## MLOps Lifecycle Highlights

### 1. Cryptographic Lineage Manifest (`lineage.json`)
Every candidate model is registered with a verifiable SHA-256 provenance hash:
```json
{
  "schema_version": "1.0.0",
  "lineage_id": "2dd29949d2a776a4bf62e5d781bb023c5c23e85b1e0f8de2d9d5361b65028117",
  "git": { "commit": "4c91398e...", "dirty": true },
  "dataset": { "archive_sha256": "0349ba38...", "sample_items": 100 },
  "features": { "count": 25, "features_hash": "64977084..." },
  "model": { "type": "lightgbm_hybrid", "version": "v2", "binary_sha256": "c8e18b99..." },
  "validation": { "horizons": [7, 28], "metrics": { "MAE": 0.9770, "WAPE": 0.6092, "Bias": -0.2701 } }
}
```

### 2. Configurable Promotion Gate
Candidates are evaluated against active models using deterministic rules defined in `pipeline.toml`:
- `min_mae_improvement_pct`: Primary metric threshold.
- `max_rmse_degradation_pct`: Zero or bounded tolerance.
- `max_wape_degradation_pct`: Volume-weighted error guardrail.
- `max_absolute_bias`: Caps systematic over/under-stocking drift.

### 3. Closed-Loop Delayed Actuals Reconciler
The `ActualsReconciler` (`src/monitoring/reconciler.py`) ingests real-world sales as they arrive, matches them against prior predictions in SQLite, and computes rolling operational metrics (`MAE`, `RMSE`, `WAPE`, `Bias`) alongside distribution PSI drift to trigger automated candidate retraining.

---

## Repository Structure

```
├── .github/workflows/ci.yml      # Automated GitHub Actions test pipeline
├── artifacts/                    # Generated benchmark comparison and drift reports
├── data/                         # Raw M5 ingestion and processing scripts
├── features/                     # Feature builders and leakage verification
├── inference/                    # FastAPI app, router, logger, and feature builder
├── monitoring/                   # SQLite prediction store and PSI drift engine
├── registry/                     # Versioned models (v1, v2) and active pointer
├── src/                          # Core Enterprise platform modules
│   ├── evaluation/               # Zero-safe metrics (MAE, RMSE, WAPE, Bias, Slices)
│   ├── lineage/                  # SHA-256 cryptographic lineage engine
│   ├── monitoring/               # Delayed actuals reconciler & operational evaluation
│   └── registry/                 # Central ModelRegistry manager & promotion gate
├── tests/                        # 27 automated unit and lifecycle tests
├── pipeline.toml                 # Production 3,049-item configuration
├── pipeline.dev.toml             # Development 100-item configuration
├── production_pipeline.py        # Production entry point and backtesting arena
├── streaming_pipeline.py         # Memory-bounded full-data streaming engine
├── pyproject.toml                # Project metadata & pytest configuration
└── requirements.txt              # Pinned core runtime dependencies
```

---

## License

This project is licensed under the MIT License.
