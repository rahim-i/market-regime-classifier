"""Central settings for the project.

Keeping every 'magic number' in one file makes it easy to see (and change)
the assumptions behind the results.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
DATA_FILE = DATA_DIR / "market_data.csv"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

# --- Data ------------------------------------------------------------------
START_DATE = "1990-01-01"
TICKERS = {
    "^GSPC": "sp500",        # S&P 500 price index
    "^SP500TR": "sp500_tr",  # S&P 500 total return index (includes dividends)
    "^VIX": "vix",           # CBOE Volatility Index ("fear gauge")
    "^IRX": "tbill",         # 13-week US Treasury bill yield, in per cent
}

# --- Regime model (Hidden Markov Model) ------------------------------------
N_STATES = 3
REGIME_NAMES = ["Calm", "Volatile", "Stress"]   # ordered from low to high volatility
HMM_FEATURES = ["ret", "log_rvol", "log_vix"]   # what the HMM "sees" each month
HMM_MIN_TRAIN = 120      # months of history before the first regime call (10 years)
HMM_SEEDS = (0, 1, 2, 3, 4)  # several random starts; keep the best fit
HMM_N_ITER = 200

# --- Prediction ------------------------------------------------------------
PRED_MIN_TRAIN = 60      # months of regime history before the first forecast (5 years)

# --- Strategy --------------------------------------------------------------
# Share of the portfolio in the S&P 500 for each regime; the rest sits in T-bills.
REGIME_WEIGHTS = {"Calm": 1.0, "Volatile": 0.5, "Stress": 0.0}
COST_BPS = 10            # transaction cost per 100% of portfolio traded (0.10%)
COST_SENSITIVITY_BPS = (0, 10, 25, 50)
