"""Detect market regimes with a 3-state Gaussian Hidden Markov Model (HMM).

The idea in one paragraph
-------------------------
An HMM assumes the market is always in one of a few hidden 'states'. Each state
produces data with its own typical return and volatility (a Gaussian, i.e. a
bell curve), and the market tends to stay in its current state, occasionally
jumping to another. We never see the state directly; the model infers it.

How we avoid look-ahead bias
----------------------------
At each month-end t we fit a *fresh* HMM using only months 1..t (an expanding
window) and ask: 'given everything up to now, how likely is each state *now*?'
That is the *filtered* probability. We never use months after t.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from . import config

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def fit_hmm(X: np.ndarray, seeds=config.HMM_SEEDS, n_iter=config.HMM_N_ITER) -> GaussianHMM:
    """Fit several HMMs from different random starting points and keep the best.

    HMM fitting can get stuck in a poor solution depending on where it starts,
    so trying a few starts and keeping the highest likelihood is standard practice.
    """
    best, best_score = None, -np.inf
    for seed in seeds:
        model = GaussianHMM(
            n_components=config.N_STATES,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X)
                score = model.score(X)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if np.isfinite(score) and score > best_score:
            best, best_score = model, score
    if best is None:
        raise RuntimeError("HMM failed to fit for every random seed.")
    return best


def state_order(model: GaussianHMM, vol_col: int) -> np.ndarray:
    """Return state indices sorted from lowest to highest mean volatility.

    The HMM numbers its states arbitrarily (state 0 might be 'Stress' in one fit
    and 'Calm' in the next). Sorting by volatility gives every state a stable,
    meaningful name: Calm < Volatile < Stress.
    """
    return np.argsort(model.means_[:, vol_col])


def filtered_regimes(
    features: pd.DataFrame,
    cols=config.HMM_FEATURES,
    min_train: int = config.HMM_MIN_TRAIN,
    seeds=config.HMM_SEEDS,
) -> pd.DataFrame:
    """Real-time (filtered) regime probabilities using an expanding window.

    Returns one row per month from month `min_train` onwards, with columns
    p_calm, p_volatile, p_stress, regime (0/1/2) and regime_name.
    """
    data = features[cols].to_numpy()
    vol_col = list(cols).index("log_rvol")
    rows = []
    for t in range(min_train - 1, len(data)):
        past = data[: t + 1]  # months 0..t only: nothing from the future
        # Standardise with statistics from the past only.
        mu, sd = past.mean(axis=0), past.std(axis=0)
        X = (past - mu) / sd
        model = fit_hmm(X, seeds=seeds)
        # predict_proba runs forward-backward over months 0..t. For the *last*
        # month there is no 'later' data, so its value equals the filtered
        # probability P(state_t | data up to t).
        probs_now = model.predict_proba(X)[-1]
        rows.append(probs_now[state_order(model, vol_col)])

    out = pd.DataFrame(
        rows,
        index=features.index[min_train - 1:],
        columns=[f"p_{n.lower()}" for n in config.REGIME_NAMES],
    )
    out["regime"] = out.to_numpy().argmax(axis=1)
    out["regime_name"] = out["regime"].map(dict(enumerate(config.REGIME_NAMES)))
    return out


def regime_summary(features: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Average behaviour of the market within each detected regime."""
    df = features.join(regimes[["regime_name"]], how="inner")
    summary = df.groupby("regime_name").agg(
        months=("ret", "size"),
        avg_monthly_return_pct=("ret", lambda r: 100 * r.mean()),
        avg_realised_vol_pct=("rvol", lambda v: 100 * v.mean()),
        avg_vix=("vix", "mean"),
    )
    summary["share_of_time_pct"] = 100 * summary["months"] / summary["months"].sum()
    return summary.reindex(config.REGIME_NAMES).round(2)
