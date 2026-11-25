"""Would knowing the regime have helped an investor?

The rule: at each month-end, look at the regime the model detects *today*
and set next month's share in the S&P 500 (the rest goes into T-bills):
Calm -> 100%, Volatile -> 50%, Stress -> 0%. Every change in the share costs
a transaction fee. Buy-and-hold simply stays 100% invested throughout.
"""

import numpy as np
import pandas as pd

from . import config


def run_strategy(weights: pd.Series, monthly: pd.DataFrame, cost_bps: float = config.COST_BPS):
    """Monthly returns of a strategy.

    weights : share in shares decided at month-end t (applied during month t+1)
    monthly : must contain 'tr_ret' (S&P 500 total return) and 'rf' (cash return)
    """
    # Decided at the end of month t, so it earns month t+1's return.
    w = weights.reindex(monthly.index).shift(1)
    df = monthly[["tr_ret", "rf"]].join(w.rename("w"), how="inner").dropna()
    turnover = df["w"].diff().abs()
    turnover.iloc[0] = df["w"].iloc[0]  # buying in at the start also costs
    gross = df["w"] * df["tr_ret"] + (1 - df["w"]) * df["rf"]
    net = gross - turnover * cost_bps / 10_000
    return pd.DataFrame({"return": net, "weight": df["w"], "turnover": turnover})


def performance(result: pd.DataFrame, rf: pd.Series) -> dict:
    r = result["return"]
    growth = (1 + r).cumprod()
    years = len(r) / 12
    excess = r - rf.reindex(r.index)
    drawdown = growth / growth.cummax() - 1
    return {
        "annual_return_pct": 100 * (growth.iloc[-1] ** (1 / years) - 1),
        "annual_volatility_pct": 100 * r.std() * np.sqrt(12),
        "sharpe_ratio": excess.mean() / excess.std() * np.sqrt(12),
        "max_drawdown_pct": 100 * drawdown.min(),
        "avg_share_in_stocks_pct": 100 * result["weight"].mean(),
        "trades_per_year": (result["turnover"] > 0).sum() / years,
        "final_value_of_1": growth.iloc[-1],
    }


def run_backtests(monthly: pd.DataFrame, regimes: pd.DataFrame):
    """Compare buy-and-hold, the regime strategy and a 'perfect hindsight' version."""
    names = regimes["regime_name"]
    regime_w = names.map(config.REGIME_WEIGHTS)
    strategies = {
        "Buy and hold": pd.Series(1.0, index=names.index),
        "Regime-aware": regime_w,
        # Not achievable in real life: uses NEXT month's regime. Shows the upper
        # limit of what regime knowledge could be worth.
        "Perfect hindsight (not achievable)": regime_w.shift(-1),
    }
    results = {k: run_strategy(w, monthly) for k, w in strategies.items()}
    # Compare all strategies over exactly the same months.
    common = results["Buy and hold"].index
    for res in results.values():
        common = common.intersection(res.index)
    results = {k: v.loc[common] for k, v in results.items()}

    table = pd.DataFrame(
        {k: performance(v, monthly["rf"]) for k, v in results.items()}
    ).T.round(2)

    # How sensitive is the regime strategy to trading costs?
    sens = pd.DataFrame({
        f"{c} bps": performance(run_strategy(regime_w, monthly, c).loc[common], monthly["rf"])
        for c in config.COST_SENSITIVITY_BPS
    }).T[["annual_return_pct", "sharpe_ratio", "max_drawdown_pct"]].round(2)
    sens.index.name = "cost per trade"
    return results, table, sens
