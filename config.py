# ==================== File: config.py ====================
"""
Configuration for P2-ETF-NTS-NOTEARS.
"""
import os

# Hugging Face configuration
HF_INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_INPUT_FILE = "master_data.parquet"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-nts-notears-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Universes
FI_COMMODITY_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "XME"]
COMBINED_TICKERS = FI_COMMODITY_TICKERS + EQUITY_TICKERS

BENCHMARK_FI = "AGG"
BENCHMARK_EQ = "SPY"

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_TRAIN_DAYS = 252 * 2
MIN_TEST_DAYS = 63
TRADING_DAYS_PER_YEAR = 252

# Change Point Detection (for adaptive window)
CP_PENALTY = 3.0
CP_MODEL = "l2"
CP_MIN_DAYS_BETWEEN = 20
CP_CONSENSUS_FRACTION = 0.5

# NTS‑NOTEARS hyperparameters
N_LAGS = 5                    # number of lagged time steps (t‑1 ... t‑5)
HIDDEN_DIMS = [32, 16]        # hidden layer sizes for MLP (excluding input/output)
LAMBDA1 = [0.02] * (N_LAGS + 1)   # L1 penalty per time step (lags + instantaneous)
LAMBDA2 = 0.01                # L2 penalty
W_THRESHOLD = [0.3] * (N_LAGS + 1)  # edge threshold for adjacency
MAX_ITER = 200                # dual ascent iterations
H_TOL = 1e-8
RHO_MAX = 1e16
DEVICE = "cpu"
