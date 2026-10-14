"""
Charts for the ALM project. Every function takes the objects or DataFrames
main.py builds and saves a PNG to output/. Nothing is hardcoded: change the
bonds, liabilities, curve or portfolios in main.py and the charts follow.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from config import OUTPUT_DIR


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Categorical colors in fixed order: series i always gets SERIES[i].
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LIABILITY_COLOR = "#8a8984"
SHORTER_COLOR = "#9cc1ee"   # light step of the blue ramp
LONGER_COLOR = "#1f5fae"    # dark step of the blue ramp
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "font.size": 9,
})


def _dollars(x, _pos=None):
    """Compact dollar labels: $1.2M, $350K, $800."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.0f}K"
    return f"{sign}${x:.0f}"


def _save(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _annual_buckets(times, amounts, horizon):
    """Sum cash flows into years 1..horizon (a flow at t lands in year ceil(t))."""
    years = np.arange(1, horizon + 1)
    totals = np.zeros(horizon)
    for t, a in zip(times, amounts):
        totals[min(max(math.ceil(t - 1e-9), 1), horizon) - 1] += a
    return years, totals


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_durations(metrics_df, liability_name, filename="durations.png"):
    """
    Effective duration of every instrument in metrics_df, with the liability
    duration as a reference line. Bonds below the line can be the short leg
    of a barbell; bonds above it can be the long leg.
    """
    bonds = metrics_df[metrics_df["name"] != liability_name].sort_values("effective")
    d_liab = metrics_df.loc[metrics_df["name"] == liability_name, "effective"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 0.32 * len(bonds) + 1.2))
    colors = [SHORTER_COLOR if d < d_liab else LONGER_COLOR for d in bonds["effective"]]
    ax.barh(bonds["name"], bonds["effective"], color=colors, height=0.6)
    ax.axvline(d_liab, color=TEXT_PRIMARY, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Effective duration (years)")
    ax.set_title("Bond durations vs. liability duration")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=SHORTER_COLOR),
                 plt.Rectangle((0, 0), 1, 1, color=LONGER_COLOR),
                 plt.Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1.5, linestyle="--")],
        labels=["Shorter than liabilities", "Longer than liabilities",
                f"Liability duration ({d_liab:.2f})"],
        loc="lower right",
    )
    return _save(fig, filename)


def plot_cash_flows(portfolios, liabilities, filename="cash_flows.png"):
    """
    Annual asset cash flows for each portfolio (one panel each) against the
    liability cash flows. Same PV and DV01 can come from very different shapes.
    """
    horizon = int(math.ceil(max(
        max(liabilities.times), *(max(p.times) for p in portfolios)
    )))
    years, liab = _annual_buckets(liabilities.times, liabilities.amounts, horizon)

    n = len(portfolios)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), sharex=True, squeeze=False)

    width = 0.42
    for i, (ax, p) in enumerate(zip(axes.flat, portfolios)):
        _, assets = _annual_buckets(p.times, p.amounts, horizon)
        ax.bar(years - width / 2, assets, width=width, color=SERIES[i % len(SERIES)], label=p.name)
        ax.bar(years + width / 2, liab, width=width, color=LIABILITY_COLOR, label=liabilities.name)
        ax.set_title(p.name)
        ax.yaxis.set_major_formatter(FuncFormatter(_dollars))
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Year")

    fig.suptitle("Asset vs. liability cash flows", color=TEXT_PRIMARY, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename)


def plot_dv01_match(portfolio_metrics_df, liability_dv01, filename="dv01_match.png"):
    """DV01 of each portfolio against the liability DV01 target."""
    df = portfolio_metrics_df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df) + 1.4))
    ax.barh(df["name"], df["dv01"],
            color=[SERIES[i % len(SERIES)] for i in range(len(df))], height=0.55)
    ax.axvline(liability_dv01, color=TEXT_PRIMARY, linewidth=1.5, linestyle="--")
    for y, v in enumerate(df["dv01"]):
        gap = v - liability_dv01
        ax.text(max(v, liability_dv01), y, f"  ${v:,.0f}  (gap {gap:+,.0f})",
                va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.set_xlabel("DV01 ($ per 1bp)")
    ax.set_title("Portfolio DV01 vs. liability DV01")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(df["dv01"].max(), liability_dv01) * 1.35)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
        handles=[plt.Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1.5, linestyle="--")],
        labels=[f"Liability DV01 (${liability_dv01:,.0f})"],
    )
    return _save(fig, filename)


def plot_surplus_vs_shock(shock_df, filename="surplus_vs_shock.png"):
    """
    Surplus change under each parallel shock, one line per portfolio.
    Flat near zero = hedged. Curvature at large shocks = convexity mismatch.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)

    for i, (name, g) in enumerate(shock_df.groupby("portfolio", sort=False)):
        g = g.sort_values("shock_bp")
        color = SERIES[i % len(SERIES)]
        ax.plot(g["shock_bp"], g["surplus_change"], color=color, linewidth=2,
                marker="o", markersize=4, label=name)

    ax.yaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.set_xlabel("Parallel shock (bp)")
    ax.set_ylabel("Change in surplus")
    ax.set_title("Surplus change under parallel rate shocks")
    ax.legend(loc="best")
    return _save(fig, filename)


def plot_approximation_error(shock_df, filename="duration_approx_error.png"):
    """
    Actual minus DV01-predicted surplus change. What remains after the
    first-order hedge: second-order (convexity) effects.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)

    for i, (name, g) in enumerate(shock_df.groupby("portfolio", sort=False)):
        g = g.sort_values("shock_bp")
        ax.plot(g["shock_bp"], g["approx_error"], color=SERIES[i % len(SERIES)],
                linewidth=2, marker="o", markersize=4, label=name)

    ax.yaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.set_xlabel("Parallel shock (bp)")
    ax.set_ylabel("Actual − DV01-predicted change")
    ax.set_title("What duration misses: error of the first-order hedge")
    ax.legend(loc="best")
    return _save(fig, filename)