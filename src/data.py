"""Download market data once and keep a raw snapshot on disk.

Why a snapshot? Prices get revised and data providers change. Saving the exact
data used means anyone who clones the repository gets exactly the same results.
"""

from datetime import date

import pandas as pd

from . import config


def download_data(start: str = config.START_DATE) -> pd.DataFrame:
    """Download daily closing values for every ticker in config.TICKERS."""
    import yfinance as yf  # imported here so tests do not need internet access

    raw = yf.download(
        list(config.TICKERS), start=start, auto_adjust=False, progress=False
    )["Close"]
    df = raw.rename(columns=config.TICKERS)[list(config.TICKERS.values())]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    # Drop days where the stock market itself has no price (e.g. stray holidays).
    return df.dropna(subset=["sp500"])


def load_data(refresh: bool = False) -> pd.DataFrame:
    """Load the saved snapshot, downloading it first if it does not exist yet.

    Pass refresh=True (or run `python run_all.py --refresh`) to replace the
    snapshot with fresh data.
    """
    if refresh or not config.DATA_FILE.exists():
        print("Downloading data from Yahoo Finance ...")
        df = download_data()
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.DATA_FILE, float_format="%.6f")
        (config.DATA_DIR / "SNAPSHOT_INFO.txt").write_text(
            f"Downloaded from Yahoo Finance via yfinance on {date.today():%d %B %Y}.\n"
            f"Tickers: {', '.join(config.TICKERS)}\n"
            f"Rows: {len(df)}, from {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}\n"
        )
    df = pd.read_csv(config.DATA_FILE, index_col="date", parse_dates=True)
    return df
