# Smart Spot Trading System: Dual-Agent Architecture
**Date:** 2025-12-12
**Status:** Hybrid Deployment (Legacy + Next-Gen)

This document serves as the **Master Context** for the Trading System, which now supports two distinct algorithmic approaches running in parallel.

---

## 🏗️ System Overview

The system is containerized using Docker and orchestrates two independent trading bots:

1.  **Agent 1 ("The Scalper")**: A legacy Supervised Learning model operating on **1-minute** candles. Use for high-frequency internal simulation.
2.  **Agent 1H ("The Sniper")**: A Next-Gen Contrastive Learning model operating on **1-hour** candles. Deployed to **Binance Testnet**.

| Feature | **Agent 1 (Legacy)** | **Agent 1H (Contrastive)** |
| :--- | :--- | :--- |
| **Philosophy** | "Predict next minute close" | "Identify Bullish Regimes & Trends" |
| **Timeframe** | `1m` (Minute) | `1h` (Hour) |
| **Core Model** | Transformer (Supervised) | Contrastive Encoder + HMM |
| **Input Data** | OHLCV (Single Asset) | OHLCV + Cross-Asset (ETH) + FFD |
| **Execution** | Internal DB Simulation | Binance Testnet (Paper Money) |
| **Status** | Maintenance / Reference | **Active Production** |

---

## 📂 File Map & Configuration

Each agent has a dedicated "Swimlane" in the codebase to prevent conflicts.

### 1. Agent 1 (Legacy)
*   **Entry Point**: `scripts/run_bot.py`
*   **Docker Service**: `bot`
*   **Model Directory**: `models/agent1/`
    *   `agent1.onnx` (Inference Model)
    *   `pca.pkl` (Feature Reducer)
    *   `hmm_regime.pkl` (Regime Detector 3-State)
*   **Config Behavior**: Uses default `config.py` settings (`TIMEFRAME=1m`).

### 2. Agent 1H (Contrastive)
*   **Entry Point**: `scripts/run_paper_1h.py`
*   **Docker Service**: `paper_trader`
*   **Model Directory**: `models/contrastive_1h/`
    *   `model_contrastive_1h.pt` (PyTorch Model)
    *   `pca_1h.pkl` (Feature Reducer 1H)
    *   `hmm_contrastive_1h.pkl` (Regime Detector 2-State)
*   **Config Behavior**: Overrides config at runtime (`config.TIMEFRAME = '1h'`).

---

## 🚀 Execution Guide

### Running Agent 1H (Production)
The main bot monitoring the hourly market.
```bash
docker compose up -d paper_trader
```

### Running Agent 1 (Legacy Side-Car)
To compare performance in a simulated environment.
```bash
# Ensure models are in models/agent1/
docker compose up -d bot
```

### Dashboard
Visualizes the database (where Agent 1 writes) and potentially logs.
```bash
docker compose up -d dashboard
```
Access at: `http://localhost:8501`

---

## 🔧 Environment Variables (.env)
Critical keys for operation.

```ini
# Exchange
BINANCE_TESTNET_KEY=...    # Required for Agent 1H
BINANCE_TESTNET_SECRET=... # Required for Agent 1H

# General
TIMEFRAME=1h               # Default (Overridden by Agent 1)
IS_PAPER_TRADING=True      # Safety Lock
```

## ⚠️ Known Quirks
*   **Agent 1H Startup**: Runs a "Turbo Mode" check (10s loop) on start, then settles into hourly schedule (`:02` minute).
*   **Data Ingestion**: Agent 1H requires ~1000 candles history for Fractional Differencing (FFD).
*   **Sklearn Limit**: Docker environment requires `reset_index(drop=True)` before PCA transformation to avoid shape errors.
