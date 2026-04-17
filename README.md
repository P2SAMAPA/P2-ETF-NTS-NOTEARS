# ==================== File: README.md ====================
# P2 ETF NTS‑NOTEARS Engine

**Nonlinear causal DAG discovery via NOTEARS‑MLP with CNN‑based temporal modelling.**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-NTS-NOTEARS/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-NTS-NOTEARS/actions/workflows/daily_run.yml)

## Overview

This engine learns a temporal causal graph among ETFs using the NTS‑NOTEARS algorithm. It models both instantaneous (intra‑day) and lagged (inter‑day) causal effects, enforcing acyclicity via a smooth continuous constraint. The learned model is then used to predict next‑day returns and select the most promising ETF.

**Key Features:**
- **Causal Discovery**: Identifies directional causal relationships (parents) for each ETF.
- **Temporal DAG**: Captures both instantaneous and time‑lagged dependencies.
- **Three Universes**: FI/Commodities, Equity Sectors, and Combined.
- **Global & Adaptive Training**: Fixed 80/10/10 split and post‑change‑point adaptive windows.
- **Daily Output**: Top ETF pick, predicted return, and causal graph information.

## Data

- **Input**: `P2SAMAPA/fi-etf-macro-signal-master-data` (master_data.parquet)
- **Output**: `P2SAMAPA/p2-etf-nts-notears-results`

## Usage

```bash
pip install -r requirements.txt
python trainer.py          # Runs training and pushes to HF
streamlit run streamlit_app.py
Configuration
All parameters are in config.py:

N_LAGS: number of lagged time steps (default 5)

HIDDEN_DIMS: MLP hidden layer sizes

LAMBDA1, LAMBDA2: L1/L2 penalties for sparsity

MAX_ITER: dual‑ascent iterations
