"""Day 14: Advanced Visualization with Seaborn - Epic 3.

Dependency: builds directly on Day 13's Matplotlib fundamentals (Seaborn
IS matplotlib underneath - every function here still returns/uses a
matplotlib Figure/Axes, just with Seaborn's higher-level statistical
plotting on top) and on Day 12's `pearson_correlation` for the heatmap
and regression-plot correlation values.

`sns.set_theme(...)` is called once at import time, per the task's
instruction to apply "a consistent theme/style across all charts for
the rest of the training" - any module that imports this one inherits
the same look.

Matplotlib (Day 13) vs. Seaborn (Day 14) - the practical difference:
    Matplotlib is the low-level drawing layer - every tick, color, and
    legend entry is set by hand, which gives full control but takes more
    code for anything statistical. Seaborn sits on top of Matplotlib and
    knows about DataFrames and statistics directly - `sns.violinplot`,
    for instance, computes and draws a kernel density estimate per
    category in one call, where doing the equivalent in raw Matplotlib
    would mean manually computing each category's density curve first.
    The tradeoff is less fine-grained control in exchange for far less
    code and a more polished, consistent default look - see
    `plot_matplotlib_vs_seaborn_comparison` below for a direct side by
    side on the same data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from day11_descriptive_stats import _load_merged_dataset
from day12_correlation_distributions import pearson_correlation
from day13_matplotlib_charts import CHARTS_DIR, _ensure_parent_dir, plot_boxplot_by_category

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Applied once, at import time - every chart made after this line (in this
# module or any that imports it) shares the same consistent look.
sns.set_theme(style="whitegrid", palette="deep")

DPI = 150
ID_COLUMN_SUFFIX = "_id"


def _numeric_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> List[str]:
    """Pick numeric, non-ID columns - same exclusion rule Day 12 uses, so the
    heatmap/pairplot cover the same columns Day 12's correlation report did."""
    if columns is not None:
        return list(columns)
    return [
        col for col in df.select_dtypes(include=[np.number]).columns
        if not col.lower().endswith(ID_COLUMN_SUFFIX)
    ]


# --------------------------------------------------------------------------
# 1. Correlation heatmap - reuses Day 12's pearson_correlation
# --------------------------------------------------------------------------
def build_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Build a square Pearson correlation matrix using Day 12's from-scratch function.

    This is the same pairwise math `correlation_matrix_report` (Day 12)
    computes, reshaped into a square DataFrame suitable for `sns.heatmap`
    instead of a long-format report table.

    Args:
        df: Input DataFrame.
        columns: Which columns to include. Defaults to all numeric,
            non-ID columns.

    Returns:
        Square DataFrame (columns == index) of Pearson correlation
        coefficients, with 1.0 on the diagonal.
    """
    cols = _numeric_columns(df, columns)
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for col_a in cols:
        for col_b in cols:
            if col_a == col_b:
                matrix.loc[col_a, col_b] = 1.0
            else:
                x = df[col_a].to_numpy(dtype=float)
                y = df[col_b].to_numpy(dtype=float)
                matrix.loc[col_a, col_b] = pearson_correlation(x, y)

    return matrix


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str, columns: Optional[List[str]] = None) -> None:
    """Plot an annotated Seaborn heatmap of the Pearson correlation matrix.

    Cell values are printed directly on the heatmap (`annot=True`), and
    the matrix itself comes from `build_correlation_matrix`, so this
    always matches Day 12's from-scratch Pearson values exactly - not a
    separately-computed `df.corr()`.

    Args:
        df: Input DataFrame.
        save_path: Where to save the .png.
        columns: Which numeric columns to include (default: all
            numeric, non-ID columns).
    """
    matrix = build_correlation_matrix(df, columns)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"label": "Pearson r"}, ax=ax,
    )
    ax.set_title("Correlation Heatmap (Pearson r, from Day 12)")
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved correlation heatmap: %s", save_path)


# --------------------------------------------------------------------------
# 2. Pairplot - pairwise scatter/KDE grid, colored by category
# --------------------------------------------------------------------------
def plot_pairplot(df: pd.DataFrame, hue_col: str, save_path: str, columns: Optional[List[str]] = None) -> None:
    """Plot a Seaborn pairplot of numeric columns, colored by `hue_col`.

    Diagonal panels show each column's histogram per category;
    off-diagonal panels show pairwise scatter plots, all colored by
    `hue_col` so category separation is visible at a glance. Histograms
    (rather than KDE) are used on the diagonal because at least one
    column here (`standard_discount_pct`) takes only a handful of fixed
    values per category, which breaks a kernel density estimate.

    Args:
        df: Input DataFrame.
        hue_col: Categorical column used to color points/densities.
        save_path: Where to save the .png.
        columns: Which numeric columns to include (default: all
            numeric, non-ID columns - kept short on purpose, since a
            pairplot's size grows with the square of the column count).
    """
    numeric_cols = _numeric_columns(df, columns)
    subset = df[numeric_cols + [hue_col]].dropna()

    grid = sns.pairplot(subset, hue=hue_col, diag_kind="hist", corner=True, height=2.2)
    grid.figure.suptitle("Pairwise Relationships by " + hue_col, y=1.02)

    _ensure_parent_dir(save_path)
    grid.savefig(save_path, dpi=DPI)
    plt.close(grid.figure)
    logger.info("Saved pairplot: %s", save_path)


# --------------------------------------------------------------------------
# 3. Violin plot - distribution + density by category
# --------------------------------------------------------------------------
def plot_violin_by_category(df: pd.DataFrame, num_col: str, cat_col: str, save_path: str) -> None:
    """Plot a Seaborn violin plot: distribution shape (via KDE) of `num_col` per `cat_col`.

    A violin plot shows everything a boxplot shows (median, quartiles)
    PLUS the full estimated density shape - e.g. it reveals a bimodal
    (two-humped) distribution within a category that a boxplot would
    flatten into a single box.

    Args:
        df: Input DataFrame.
        num_col: Numeric column to summarize.
        cat_col: Categorical column to group by.
        save_path: Where to save the .png.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    working = df[[num_col, cat_col]].dropna()
    if working.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{num_col} by {cat_col} (no data)")
    else:
        order = sorted(working[cat_col].unique())
        sns.violinplot(data=working, x=cat_col, y=num_col, order=order, hue=cat_col, legend=False, ax=ax)
        ax.set_title(f"Distribution of {num_col} by {cat_col}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved violin plot: %s", save_path)


# --------------------------------------------------------------------------
# 4. Regression scatter - correlation with a fitted line
# --------------------------------------------------------------------------
def plot_regression_scatter(df: pd.DataFrame, x_col: str, y_col: str, save_path: str) -> None:
    """Plot a Seaborn regression scatter (`sns.regplot`): points plus a fitted linear regression line with confidence band.

    The title reports the same Pearson r Day 12's `pearson_correlation`
    computes, so the visual fit and the numeric statistic always agree.

    Args:
        df: Input DataFrame.
        x_col: Column for the x-axis.
        y_col: Column for the y-axis.
        save_path: Where to save the .png.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    working = df[[x_col, y_col]].dropna()
    if working.shape[0] < 2:
        ax.text(0.5, 0.5, "Not enough data to plot", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{y_col} vs {x_col} (no data)")
    else:
        sns.regplot(
            data=working, x=x_col, y=y_col, ax=ax,
            scatter_kws={"alpha": 0.6, "edgecolor": "black"},
            line_kws={"color": "crimson"},
        )
        try:
            r = pearson_correlation(working[x_col].to_numpy(dtype=float), working[y_col].to_numpy(dtype=float))
            ax.set_title(f"{y_col} vs {x_col} with fitted regression line (Pearson r = {r:+.3f})")
        except ValueError:
            ax.set_title(f"{y_col} vs {x_col} with fitted regression line (r undefined)")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved regression scatter: %s", save_path)


# --------------------------------------------------------------------------
# 5. Bonus: countplot for categorical frequency
# --------------------------------------------------------------------------
def plot_countplot(df: pd.DataFrame, cat_col: str, save_path: str) -> None:
    """Plot a Seaborn countplot: frequency of each category in `cat_col`, ordered descending.

    Args:
        df: Input DataFrame.
        cat_col: Categorical column to count.
        save_path: Where to save the .png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if df.empty or cat_col not in df.columns:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"{cat_col} (no data)")
    else:
        order = df[cat_col].value_counts().index
        sns.countplot(data=df, x=cat_col, order=order, hue=cat_col, legend=False, ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fontsize=8)
        ax.set_title(f"Frequency of {cat_col}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel(cat_col)
    ax.set_ylabel("Count")
    fig.tight_layout()

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved countplot: %s", save_path)


# --------------------------------------------------------------------------
# 6. Matplotlib vs. Seaborn - side-by-side comparison
# --------------------------------------------------------------------------
def plot_matplotlib_vs_seaborn_comparison(df: pd.DataFrame, num_col: str, cat_col: str, save_path: str) -> None:
    """Plot Day 13's Matplotlib boxplot next to Seaborn's violin plot, on the same data.

    Styling/readability differences observed (see module docstring for
    the general Matplotlib-vs-Seaborn tradeoff):
        - Seaborn's default theme (gridlines, muted palette) reads as
          more polished immediately, with zero styling code, vs. the
          hand-picked colors/edgecolors Day 13's Matplotlib version needed.
        - The violin plot shows the full density SHAPE per category
          (e.g. whether a category's prices cluster or spread evenly),
          information the boxplot's five-number summary alone doesn't
          show - at the cost of being a less familiar chart to a general
          audience than a boxplot.
        - Both agree on the median/IQR story; the violin just adds the
          density shape on top.

    Args:
        df: Input DataFrame.
        num_col: Numeric column to summarize.
        cat_col: Categorical column to group by.
        save_path: Where to save the combined .png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{num_col} by {cat_col}: Matplotlib (Day 13) vs. Seaborn (Day 14)", fontsize=14, fontweight="bold")

    # Left: Day 13's matplotlib boxplot logic, drawn directly onto axes[0]
    # (re-implemented inline rather than calling plot_boxplot_by_category,
    # since that function saves its own file rather than plotting to a
    # given Axes).
    working = df[[num_col, cat_col]].dropna()
    categories = sorted(working[cat_col].unique())
    grouped_values = [working.loc[working[cat_col] == cat, num_col].to_numpy(dtype=float) for cat in categories]
    box = axes[0].boxplot(grouped_values, tick_labels=categories, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#8CB369")
        patch.set_alpha(0.7)
    axes[0].set_title("Matplotlib: boxplot (Day 13 style)")
    axes[0].set_xlabel(cat_col)
    axes[0].set_ylabel(num_col)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    # Right: Seaborn violin plot, same data
    sns.violinplot(data=working, x=cat_col, y=num_col, order=categories, hue=cat_col, legend=False, ax=axes[1])
    axes[1].set_title("Seaborn: violinplot (Day 14 style)")
    axes[1].set_xlabel(cat_col)
    axes[1].set_ylabel(num_col)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    logger.info("Saved Matplotlib-vs-Seaborn comparison: %s", save_path)


# --------------------------------------------------------------------------
# 7. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 14: Advanced Visualization with Seaborn ===")

    df = _load_merged_dataset()
    logger.info("Loaded merged Epic 2 dataset: %s", df.shape)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_correlation_heatmap(df, str(CHARTS_DIR / "seaborn_heatmap_correlation.png"))
    plot_pairplot(df, "product_category", str(CHARTS_DIR / "seaborn_pairplot_by_category.png"))
    plot_violin_by_category(df, "unit_price", "product_category", str(CHARTS_DIR / "seaborn_violin_unit_price_by_category.png"))
    plot_regression_scatter(df, "quantity", "unit_price", str(CHARTS_DIR / "seaborn_regression_quantity_vs_unit_price.png"))
    plot_countplot(df, "product_category", str(CHARTS_DIR / "seaborn_countplot_category.png"))
    plot_matplotlib_vs_seaborn_comparison(df, "unit_price", "product_category", str(CHARTS_DIR / "seaborn_vs_matplotlib_comparison.png"))

    matrix = build_correlation_matrix(df)
    logger.info("Correlation matrix (matches Day 12's Pearson values):\n%s", matrix.round(4))

    saved = sorted(p.name for p in CHARTS_DIR.glob("seaborn_*.png"))
    logger.info("Seaborn charts saved to %s: %s", CHARTS_DIR, saved)


if __name__ == "__main__":
    main()
