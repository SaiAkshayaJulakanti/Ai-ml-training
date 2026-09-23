"""Day 13: Data Visualization with Matplotlib - Epic 3.

Dependency: visualizes the statistical findings from Day 11
(`compute_central_tendency`, `compute_spread`) and Day 12
(`pearson_correlation`) rather than recomputing them - e.g. the
histogram annotates the Day 11 mean, and the scatter plot's title
reports the Day 12 Pearson r for the pair being plotted.

Chart hygiene applied to every plot in this module:
    - A descriptive title (not just the column name)
    - Labeled axes
    - A legend wherever more than one visual element needs distinguishing
    - A deliberately chosen figure size (not matplotlib's tiny default)
    - Saved at 150 DPI (the task's minimum) via `savefig(..., dpi=150)`
    - `plt.close()` after every save, so repeated calls in a script or a
      test suite don't silently accumulate open figures in memory
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: charts are files, never shown inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from day11_descriptive_stats import _load_merged_dataset, compute_central_tendency, compute_spread
from day12_correlation_distributions import pearson_correlation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHARTS_DIR = Path(__file__).parent / "charts"
FIGSIZE_SINGLE = (8, 5)
DPI = 150


def _ensure_parent_dir(save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def _empty_placeholder(ax, message: str = "No data available") -> None:
    """Draw a clean 'no data' placeholder instead of crashing on empty input."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, color="gray", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------
# 1. Histogram - distribution of a numeric column
# --------------------------------------------------------------------------
def plot_histogram(data: pd.Series, column_name: str, save_path: str) -> None:
    """Plot a histogram of `data`, annotated with the Day 11 from-scratch mean.

    Handles an empty series gracefully (draws a placeholder rather than
    raising), so this is always safe to call in a batch/report pipeline.

    Args:
        data: Numeric values to plot.
        column_name: Used in the title/axis label.
        save_path: Where to save the .png (parent directories are created
            if needed).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    clean = pd.Series(data).dropna()

    if clean.empty:
        _empty_placeholder(ax)
        ax.set_title(f"Distribution of {column_name} (no data)")
    else:
        values = clean.to_numpy(dtype=float)
        mean = compute_central_tendency(values)["mean"]
        std = compute_spread(values)["std_population"] if values.size >= 2 else 0.0

        ax.hist(values, bins=min(20, max(5, values.size // 3)), color="#4C72B0", edgecolor="black", alpha=0.85)
        ax.axvline(mean, color="crimson", linestyle="--", linewidth=2, label=f"Mean = {mean:.2f}")
        if std > 0:
            ax.axvline(mean + std, color="darkorange", linestyle=":", linewidth=1.5, label=f"+/-1 std = {std:.2f}")
            ax.axvline(mean - std, color="darkorange", linestyle=":", linewidth=1.5)
        ax.set_title(f"Distribution of {column_name}")
        ax.legend()

    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved histogram: %s", save_path)


# --------------------------------------------------------------------------
# 2. Boxplot - outlier visualization per category
# --------------------------------------------------------------------------
def plot_boxplot_by_category(df: pd.DataFrame, num_col: str, cat_col: str, save_path: str) -> None:
    """Plot a boxplot of `num_col`, grouped by `cat_col` - one box per category.

    Handles a single-category DataFrame and an empty DataFrame gracefully
    (draws one box, or a placeholder, respectively - never raises).

    Args:
        df: Input DataFrame.
        num_col: Numeric column to summarize.
        cat_col: Categorical column to group by.
        save_path: Where to save the .png.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    working = df[[num_col, cat_col]].dropna()
    if working.empty:
        _empty_placeholder(ax)
        ax.set_title(f"{num_col} by {cat_col} (no data)")
    else:
        categories = sorted(working[cat_col].unique())
        grouped_values = [working.loc[working[cat_col] == cat, num_col].to_numpy(dtype=float) for cat in categories]

        box = ax.boxplot(grouped_values, tick_labels=categories, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor("#8CB369")
            patch.set_alpha(0.7)

        ax.set_title(f"{num_col} by {cat_col} (n={len(categories)} categor{'y' if len(categories) == 1 else 'ies'})")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved boxplot: %s", save_path)


# --------------------------------------------------------------------------
# 3. Scatter plot - correlation between 2 numeric columns
# --------------------------------------------------------------------------
def plot_scatter_correlation(df: pd.DataFrame, x_col: str, y_col: str, save_path: str) -> None:
    """Plot a scatter of `y_col` vs `x_col`, titled with the Day 12 Pearson r.

    Args:
        df: Input DataFrame.
        x_col: Column for the x-axis.
        y_col: Column for the y-axis.
        save_path: Where to save the .png.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    working = df[[x_col, y_col]].dropna()
    if working.shape[0] < 2:
        _empty_placeholder(ax, "Not enough data to plot")
        ax.set_title(f"{y_col} vs {x_col} (no data)")
    else:
        x = working[x_col].to_numpy(dtype=float)
        y = working[y_col].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.6, edgecolor="black", color="#C44E52", label=f"{working.shape[0]} orders")

        try:
            r = pearson_correlation(x, y)  # reuses Day 12's from-scratch function
            ax.set_title(f"{y_col} vs {x_col}  (Pearson r = {r:+.3f})")
        except ValueError:
            ax.set_title(f"{y_col} vs {x_col}  (r undefined - constant column)")
        ax.legend()

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved scatter plot: %s", save_path)


# --------------------------------------------------------------------------
# 4. Bar chart - category counts / aggregates
# --------------------------------------------------------------------------
def plot_bar_chart(df: pd.DataFrame, cat_col: str, save_path: str, value_col: Optional[str] = None) -> None:
    """Plot a bar chart: order count per category, or a sum of `value_col` per category if given.

    Args:
        df: Input DataFrame.
        cat_col: Categorical column to group by.
        save_path: Where to save the .png.
        value_col: If given, bars show the SUM of this numeric column per
            category instead of a plain row count.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    if df.empty or cat_col not in df.columns:
        _empty_placeholder(ax)
        ax.set_title(f"{cat_col} (no data)")
    else:
        if value_col:
            summary = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False)
            ylabel = f"Total {value_col}"
        else:
            summary = df[cat_col].value_counts()
            ylabel = "Order count"

        bars = ax.bar(summary.index.astype(str), summary.values, color="#55A868", edgecolor="black")
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
        ax.set_title(f"{ylabel} by {cat_col}")
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel(cat_col)
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved bar chart: %s", save_path)


# --------------------------------------------------------------------------
# 5. Line chart - trend over time
# --------------------------------------------------------------------------
def plot_line_chart(df: pd.DataFrame, date_col: str, value_col: str, save_path: str, freq: str = "ME") -> None:
    """Plot a line chart of `value_col` summed per time period (default: monthly).

    Args:
        df: Input DataFrame.
        date_col: Datetime column to group by.
        value_col: Numeric column to sum within each period.
        save_path: Where to save the .png.
        freq: Pandas offset alias for the grouping period ("ME" = month end,
            "W" = week, "D" = day).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    working = df[[date_col, value_col]].dropna()
    if working.empty:
        _empty_placeholder(ax)
        ax.set_title(f"{value_col} over time (no data)")
    else:
        working = working.copy()
        working[date_col] = pd.to_datetime(working[date_col])
        trend = working.set_index(date_col)[value_col].resample(freq).sum()

        ax.plot(trend.index, trend.values, marker="o", color="#4C72B0", linewidth=2, label=f"Total {value_col}")
        ax.set_title(f"{value_col} trend over time (by {freq})")
        ax.legend()
        fig.autofmt_xdate(rotation=45)

    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved line chart: %s", save_path)


# --------------------------------------------------------------------------
# 6. Dashboard - 2x2 subplot grid
# --------------------------------------------------------------------------
def plot_dashboard_grid(df: pd.DataFrame, save_path: str) -> None:
    """Build one figure with a 2x2 grid: histogram, boxplot, scatter, and bar chart.

    All four panels summarize the same DataFrame under one shared title,
    so the dashboard works as a single at-a-glance report image.

    Args:
        df: Input DataFrame (expects 'unit_price', 'quantity', and
            'product_category' columns, matching the Epic 2 dataset).
        save_path: Where to save the combined .png.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Epic 2 Dataset - Statistical Dashboard", fontsize=16, fontweight="bold")

    # Top-left: histogram of unit_price
    ax = axes[0, 0]
    prices = df["unit_price"].dropna().to_numpy(dtype=float)
    mean_price = compute_central_tendency(prices)["mean"]
    ax.hist(prices, bins=15, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.axvline(mean_price, color="crimson", linestyle="--", linewidth=2, label=f"Mean = {mean_price:.2f}")
    ax.set_title("Distribution of unit_price")
    ax.set_xlabel("unit_price")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Top-right: boxplot of unit_price by product_category
    ax = axes[0, 1]
    categories = sorted(df["product_category"].dropna().unique())
    grouped_values = [df.loc[df["product_category"] == cat, "unit_price"].to_numpy(dtype=float) for cat in categories]
    box = ax.boxplot(grouped_values, tick_labels=categories, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#8CB369")
        patch.set_alpha(0.7)
    ax.set_title("unit_price by product_category")
    ax.set_xlabel("product_category")
    ax.set_ylabel("unit_price")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Bottom-left: scatter of quantity vs unit_price
    ax = axes[1, 0]
    x = df["quantity"].to_numpy(dtype=float)
    y = df["unit_price"].to_numpy(dtype=float)
    ax.scatter(x, y, alpha=0.6, edgecolor="black", color="#C44E52", label=f"{len(x)} orders")
    r = pearson_correlation(x, y)
    ax.set_title(f"unit_price vs quantity (r = {r:+.3f})")
    ax.set_xlabel("quantity")
    ax.set_ylabel("unit_price")
    ax.legend()

    # Bottom-right: bar chart of order counts per category
    ax = axes[1, 1]
    counts = df["product_category"].value_counts()
    bars = ax.bar(counts.index.astype(str), counts.values, color="#55A868", edgecolor="black")
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    ax.set_title("Order count by product_category")
    ax.set_xlabel("product_category")
    ax.set_ylabel("Order count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved dashboard grid: %s", save_path)


# --------------------------------------------------------------------------
# 7. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 13: Data Visualization with Matplotlib ===")

    df = _load_merged_dataset()
    logger.info("Loaded merged Epic 2 dataset: %s", df.shape)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_histogram(df["unit_price"], "unit_price", str(CHARTS_DIR / "histogram_unit_price.png"))
    plot_boxplot_by_category(df, "unit_price", "product_category", str(CHARTS_DIR / "boxplot_unit_price_by_category.png"))
    plot_scatter_correlation(df, "quantity", "unit_price", str(CHARTS_DIR / "scatter_quantity_vs_unit_price.png"))
    plot_bar_chart(df, "product_category", str(CHARTS_DIR / "bar_orders_by_category.png"))
    plot_line_chart(df, "order_date", "unit_price", str(CHARTS_DIR / "line_unit_price_trend.png"), freq="ME")
    plot_dashboard_grid(df, str(CHARTS_DIR / "dashboard.png"))

    saved = sorted(p.name for p in CHARTS_DIR.glob("*.png"))
    logger.info("All charts saved to %s: %s", CHARTS_DIR, saved)


if __name__ == "__main__":
    main()
