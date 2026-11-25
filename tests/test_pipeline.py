"""Basic tests. Run with:  python -m pytest

They use small synthetic (made-up) data, so they run quickly and offline.
"""

import numpy as np
import pandas as pd
import pytest

from src import backtest, features, prediction, regimes


# --- Synthetic data helpers --------------------------------------------------

def synthetic_monthly(n=140, seed=0):
    """Monthly features with three clearly different volatility levels."""
    rng = np.random.default_rng(seed)
    state = np.repeat(rng.choice(3, size=n // 10 + 1), 10)[:n]  # 10-month spells
    vol = np.array([0.10, 0.18, 0.35])[state]
    df = pd.DataFrame(index=pd.date_range("1990-01-31", periods=n, freq="ME"))
    df["ret"] = rng.normal(0.01 - 0.02 * state, vol / np.sqrt(12))
    df["rvol"] = vol * np.exp(rng.normal(0, 0.1, n))
    df["log_rvol"] = np.log(df["rvol"])
    df["vix"] = 100 * vol * 1.2 * np.exp(rng.normal(0, 0.1, n))  # own noise, like real VIX
    df["log_vix"] = np.log(df["vix"])
    df["tr_ret"] = np.exp(df["ret"]) - 1
    df["rf"] = 0.002
    return df, state


def synthetic_daily(n_days=600, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    return pd.DataFrame(
        {"sp500": px, "sp500_tr": px * 1.1, "vix": 20 + rng.normal(0, 2, n_days),
         "tbill": 3.0}, index=idx
    )


# --- Tests ---------------------------------------------------------------------

def test_regimes_have_no_lookahead():
    """Adding future data must not change any past regime call."""
    monthly, _ = synthetic_monthly(n=140)
    kwargs = dict(min_train=100, seeds=(0,))
    full = regimes.filtered_regimes(monthly, **kwargs)
    truncated = regimes.filtered_regimes(monthly.iloc[:125], **kwargs)
    overlap = truncated.index
    pd.testing.assert_frame_equal(full.loc[overlap], truncated)


def test_monthly_features_have_no_lookahead():
    """Changing prices in a later month must not change earlier months' features."""
    daily = synthetic_daily()
    base = features.monthly_features(daily)
    cutoff = base.index[10]
    shocked = daily.copy()
    shocked.loc[shocked.index > cutoff, ["sp500", "sp500_tr", "vix"]] *= 3
    after = features.monthly_features(shocked)
    pd.testing.assert_frame_equal(base.loc[:cutoff], after.loc[:cutoff])


def test_walk_forward_trains_only_on_the_past():
    """At prediction row t the model may only have seen rows 0..t-1."""
    monthly, _ = synthetic_monthly(n=60)
    reg = pd.DataFrame({"p_calm": 0.5, "p_volatile": 0.3, "p_stress": 0.2,
                        "regime": [i % 3 for i in range(60)]}, index=monthly.index)
    df = prediction.build_dataset(monthly, reg)
    seen = []

    class Recorder:
        def fit(self, X, y):
            self.n = len(X)
            return self

        def predict(self, X):
            seen.append((self.n, X))
            return np.array([0.0])

    preds = prediction.walk_forward(df, "next_log_rvol", Recorder, min_train=20, task="reg")
    X_all = df[prediction.FEATURE_COLS].to_numpy()
    for (n_train, x), month in zip(seen, preds.index):
        t = df.index.get_loc(month)
        assert n_train == t                      # trained on exactly rows 0..t-1
        np.testing.assert_array_equal(x[0], X_all[t])


def test_states_sorted_calm_to_stress():
    """State labels must be ordered from lowest to highest volatility."""
    monthly, true_state = synthetic_monthly(n=200, seed=3)
    reg = regimes.filtered_regimes(monthly, min_train=199, seeds=(0,))
    summary = regimes.regime_summary(monthly, regimes.filtered_regimes(
        monthly, min_train=150, seeds=(0,)))
    vols = summary["avg_realised_vol_pct"].dropna().to_numpy()
    assert np.all(np.diff(vols) > 0)
    # On such clear data the latest call should match the true state.
    assert reg["regime"].iloc[-1] == true_state[-1]


def test_backtest_basics():
    monthly, _ = synthetic_monthly(n=40)
    ones = pd.Series(1.0, index=monthly.index)
    bh = backtest.run_strategy(ones, monthly, cost_bps=0)
    # Fully invested and no costs: returns equal the index return.
    np.testing.assert_allclose(bh["return"], monthly["tr_ret"].iloc[1:])
    # Fully in cash: returns equal the T-bill return.
    cash = backtest.run_strategy(ones * 0, monthly, cost_bps=0)
    np.testing.assert_allclose(cash["return"], monthly["rf"].iloc[1:])
    # Trading costs can only make a switching strategy worse.
    switching = pd.Series(np.tile([1.0, 0.0], 20), index=monthly.index)
    cheap = backtest.run_strategy(switching, monthly, cost_bps=0)["return"].sum()
    dear = backtest.run_strategy(switching, monthly, cost_bps=50)["return"].sum()
    assert dear < cheap
    assert dear == pytest.approx(cheap - 39 * 0.005)  # 39 trades of 100% at 0.5%
