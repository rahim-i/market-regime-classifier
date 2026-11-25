"""Predict next month's regime and volatility, using walk-forward validation.

Walk-forward validation in one paragraph
----------------------------------------
To mimic real life, we stand at each month-end in turn, train the model only
on months whose outcome is already known, predict the coming month, then step
forward one month and repeat. Every prediction is therefore 'out of sample':
the model has never seen the month it is predicting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

FEATURE_COLS = [
    "p_calm", "p_volatile", "p_stress",   # current regime probabilities
    "ret", "ret_3m",                       # recent returns
    "log_rvol", "log_vix",                 # current volatility
    "vix_minus_rvol",                      # 'fear premium': VIX above realised vol
    "d_log_vix",                           # change in VIX over the month
]


def build_dataset(features: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """One row per month: inputs known at month-end t, targets from month t+1."""
    df = features.join(regimes, how="inner")
    df["ret_3m"] = df["ret"].rolling(3).sum()
    df["vix_minus_rvol"] = df["log_vix"] - df["log_rvol"]
    df["d_log_vix"] = df["log_vix"].diff()
    # Targets: shift(-1) pulls *next* month's value onto this month's row.
    df["next_regime"] = df["regime"].shift(-1)
    df["next_log_rvol"] = df["log_rvol"].shift(-1)
    return df.dropna(subset=FEATURE_COLS)


# --- Models -----------------------------------------------------------------

def classifiers():
    return {
        "Logistic regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)
        ),
        "Gradient boosting": lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
        ),
    }


def regressors():
    return {
        "Linear regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "Gradient boosting": lambda: GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
        ),
    }


# --- Walk-forward engine ----------------------------------------------------

def walk_forward(df, target, make_model, min_train=config.PRED_MIN_TRAIN, task="class"):
    """Expanding-window walk-forward predictions.

    At row t we train on rows 0..t-1. Row t-1's target is month t's value,
    which is known by the end of month t, so no future information is used.
    """
    X = df[FEATURE_COLS].to_numpy()
    y = df[target].to_numpy()
    preds, probs, idx = [], [], []
    for t in range(min_train, len(df)):
        if np.isnan(y[t]):  # the final month has no 'next month' to check against
            continue
        X_tr, y_tr = X[:t], y[:t]
        if task == "class":
            classes = np.unique(y_tr)
            p = np.zeros(config.N_STATES)
            if len(classes) == 1:  # early on, some regimes may not have appeared yet
                p[int(classes[0])] = 1.0
            else:
                model = make_model().fit(X_tr, y_tr.astype(int))
                p[model.classes_.astype(int)] = model.predict_proba(X[t : t + 1])[0]
            probs.append(p)
            preds.append(int(p.argmax()))
        else:
            model = make_model().fit(X_tr, y_tr)
            preds.append(float(model.predict(X[t : t + 1])[0]))
        idx.append(df.index[t])
    out = pd.DataFrame({"pred": preds}, index=idx)
    if task == "class":
        out[[f"prob_{n.lower()}" for n in config.REGIME_NAMES]] = np.array(probs)
    return out


def _baselines(df, target, min_train, task):
    """Persistence ('next month = this month') and historical average."""
    rows = []
    current = "regime" if task == "class" else "log_rvol"
    for t in range(min_train, len(df)):
        y_past = df[target].iloc[:t].to_numpy()
        if np.isnan(df[target].iloc[t]):
            continue
        if task == "class":
            hist = int(pd.Series(y_past).mode().iloc[0])  # most common regime so far
        else:
            hist = float(y_past.mean())
        rows.append((df.index[t], df[current].iloc[t], hist))
    b = pd.DataFrame(rows, columns=["month", "Persistence", "Historical average"]).set_index("month")
    return b


# --- Evaluation -------------------------------------------------------------

def run_predictions(df: pd.DataFrame, min_train: int = config.PRED_MIN_TRAIN):
    """Run every model and baseline; return predictions and a metrics table."""
    # Regime classification
    reg_preds = _baselines(df, "next_regime", min_train, "class").astype(int)
    for name, make in classifiers().items():
        reg_preds[name] = walk_forward(df, "next_regime", make, min_train, "class")["pred"]
    actual_regime = df.loc[reg_preds.index, "next_regime"].astype(int)

    stress = config.N_STATES - 1
    class_rows = []
    for name in reg_preds:
        p = reg_preds[name]
        class_rows.append({
            "model": name,
            "accuracy_pct": 100 * accuracy_score(actual_regime, p),
            "balanced_accuracy_pct": 100 * balanced_accuracy_score(actual_regime, p),
            "stress_recall_pct": 100 * recall_score(
                actual_regime, p, labels=[stress], average="macro", zero_division=0
            ),
        })

    # Volatility regression (on log scale, reported in volatility points)
    vol_preds = _baselines(df, "next_log_rvol", min_train, "reg")
    for name, make in regressors().items():
        vol_preds[name] = walk_forward(df, "next_log_rvol", make, min_train, "reg")["pred"]
    actual_log_vol = df.loc[vol_preds.index, "next_log_rvol"]

    hist_sse = ((actual_log_vol - vol_preds["Historical average"]) ** 2).sum()
    vol_rows = []
    for name in vol_preds:
        p = vol_preds[name]
        sse = ((actual_log_vol - p) ** 2).sum()
        vol_rows.append({
            "model": name,
            "mae_vol_points": 100 * mean_absolute_error(np.exp(actual_log_vol), np.exp(p)),
            "r2_vs_hist_avg_pct": 100 * (1 - sse / hist_sse),
        })

    return {
        "regime_predictions": reg_preds.assign(actual=actual_regime),
        "vol_predictions": np.exp(vol_preds).assign(actual=np.exp(actual_log_vol)),
        "regime_metrics": pd.DataFrame(class_rows).set_index("model").round(1),
        "vol_metrics": pd.DataFrame(vol_rows).set_index("model").round(2),
    }
