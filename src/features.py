"""Turn daily prices into one row of features per month.

Every value in a month's row uses only data from that month or earlier.
"""

import numpy as np
import pandas as pd


def _drop_incomplete_last_month(daily: pd.DataFrame) -> pd.DataFrame:
    """Remove the final month if it has not finished yet (e.g. data ends mid-month)."""
    last = daily.index[-1]
    month_end = last + pd.offsets.BMonthEnd(0)  # last business day of that month
    if last < month_end:
        return daily[daily.index < last.replace(day=1)]
    return daily


def monthly_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build monthly features from daily data.

    Columns
    -------
    ret       : log return of the S&P 500 over the month
    rvol      : realised volatility: std. dev. of daily log returns, annualised
    log_rvol  : natural log of rvol (volatility is skewed; logs tame the spikes)
    vix, log_vix : VIX on the last trading day of the month
    tr_ret    : simple total return (with dividends) over the month, for the backtest
    rf        : cash (T-bill) return for the month, using the yield known at the
                *start* of the month
    """
    daily = _drop_incomplete_last_month(daily.copy())
    daily["tbill"] = daily["tbill"].ffill()
    daily_logret = np.log(daily["sp500"]).diff()

    month = daily.index.to_period("M")
    g = daily.groupby(month)
    out = pd.DataFrame(index=g.size().index)
    out["ret"] = daily_logret.groupby(month).sum()
    out["rvol"] = daily_logret.groupby(month).std() * np.sqrt(252)
    out["log_rvol"] = np.log(out["rvol"])
    out["vix"] = g["vix"].last()
    out["log_vix"] = np.log(out["vix"])
    out["tr_ret"] = g["sp500_tr"].last().pct_change()
    # T-bill yield at the end of last month, converted to a monthly return.
    out["rf"] = (g["tbill"].last() / 100 / 12).shift(1)

    out.index = out.index.to_timestamp(how="end").normalize()  # month-end dates
    out.index.name = "month"
    # The first month has no previous month, so its return-based values are missing.
    return out.iloc[1:]
