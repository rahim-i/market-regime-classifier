"""Charts for the README. Every chart is saved as a PNG in figures/."""

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator
from sklearn.metrics import confusion_matrix

from . import config

# Colour-blind-checked palette. Text stays in neutral ink; colour marks identity.
REGIME_COLOURS = {"Calm": "#2a78d6", "Volatile": "#eda100", "Stress": "#e34948"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False,
})


def _plain_log_axis(ax, fmt="{:,.0f}"):
    """Log scale with ordinary numbers (1,000 instead of 10^3)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _save(fig, name):
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name)
    plt.close(fig)


def plot_regimes(daily: pd.DataFrame, regimes: pd.DataFrame):
    """S&P 500 price with the background shaded by the real-time regime."""
    start = regimes.index[0].replace(day=1)
    px = daily.loc[start:, "sp500"]
    vix = daily.loc[start:, "vix"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    for month_end, name in regimes["regime_name"].items():
        for a in (ax, ax2):
            a.axvspan(month_end.replace(day=1), month_end + pd.Timedelta(days=1), color=REGIME_COLOURS[name],
                      alpha=0.28, lw=0)
    ax.plot(px.index, px, color=INK, lw=1.1)
    _plain_log_axis(ax)
    ax.set_ylabel("S&P 500 (log scale)")
    ax.set_title("S&P 500 with real-time market regimes (each month uses only past data)")
    ax.legend(handles=[Patch(color=c, alpha=0.5, label=n) for n, c in REGIME_COLOURS.items()],
              loc="upper left", ncol=3)
    ax2.plot(vix.index, vix, color=INK_2, lw=0.9)
    ax2.set_ylabel("VIX")
    ax2.set_xlim(px.index[0], px.index[-1])
    _save(fig, "regimes_price.png")


def plot_confusion(reg_preds: pd.DataFrame, models=("Persistence", "Logistic regression")):
    """Confusion matrices: rows = what actually happened, columns = what was predicted."""
    names = config.REGIME_NAMES
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4))
    for ax, m in zip(np.atleast_1d(axes), models):
        cm = confusion_matrix(reg_preds["actual"], reg_preds[m], labels=range(len(names)))
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{100 * share[i, j]:.0f}%\n({cm[i, j]})", ha="center",
                        va="center", fontsize=10, color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted regime next month")
        ax.set_ylabel("Actual regime next month")
        acc = 100 * np.trace(cm) / cm.sum()
        ax.set_title(f"{m} ({acc:.0f}% correct)")
        ax.grid(False)
    fig.suptitle("Predicting next month's regime, walk-forward out-of-sample", x=0.02,
                 ha="left", fontweight="bold", y=1.02)
    _save(fig, "confusion_matrix.png")


def plot_equity(results: dict):
    """Growth of $1 and drawdowns for each strategy."""
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    styles = ["-", "-", "--"]
    for (name, res), colour, ls in zip(results.items(), SERIES, styles):
        growth = (1 + res["return"]).cumprod()
        dd = 100 * (growth / growth.cummax() - 1)
        ax.plot(growth.index, growth, color=colour, ls=ls, label=name)
        ax.annotate(f"${growth.iloc[-1]:.1f}", (growth.index[-1], growth.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", color=INK_2)
        ax2.plot(dd.index, dd, color=colour, ls=ls, lw=1.1)
    _plain_log_axis(ax, "${:,.0f}")
    ax.set_ylabel("Value of $1 invested (log scale)")
    ax.set_title(f"Regime-aware allocation vs buy-and-hold (after {config.COST_BPS} bps trading costs)")
    ax.legend(loc="upper left")
    ax2.set_ylabel("Fall from peak (%)")
    _save(fig, "equity_curves.png")


def plot_vol_forecast(vol_preds: pd.DataFrame, model="Linear regression"):
    """Next month's realised volatility: forecast vs what actually happened."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(vol_preds.index, 100 * vol_preds["actual"], color=INK_2, lw=1.0, label="Actual")
    ax.plot(vol_preds.index, 100 * vol_preds[model], color=SERIES[0], label=f"{model} forecast")
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title("Forecasting next month's volatility, walk-forward out-of-sample")
    ax.legend(loc="upper right")
    _save(fig, "vol_forecast.png")
