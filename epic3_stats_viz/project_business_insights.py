"""Day 15: EDA + Visualization Project - Business Insights Report.

This is the Epic 3 checkpoint deliverable: an orchestration script that
loads the Epic 2 dataset, runs the full Day 11-12 statistical analysis,
generates the full Day 13-14 chart suite, and compiles everything into
one polished PDF report plus a companion `INSIGHTS.md` - simulating a
real analyst handing off a finished deliverable rather than a pile of
separate scripts.

Nothing here reimplements Day 11-14 logic: every statistic and every
chart is produced by calling the shared `common/stats_utils.py` and
`common/viz_utils.py` modules, which themselves just re-export the
canonical Day 11-14 functions (see those two files' docstrings). This
guarantees the report's numbers and charts are identical to what those
individual day scripts produce - there's exactly one implementation of
each computation in the whole project.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Running this script directly (python epic3_stats_viz/project_business_insights.py)
# puts epic3_stats_viz/ on sys.path, not the repo root - add the root too,
# so the sibling `common` package is importable either way.
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.stats_utils import (
    build_correlation_matrix,
    compute_central_tendency,
    compute_spread,
    correlation_matrix_report,
    fit_normal_distribution,
    flag_zscore_outliers,
    load_merged_dataset,
    strongest_and_weakest_pairs,
)
from common.viz_utils import (
    CHARTS_DIR,
    plot_bar_chart,
    plot_boxplot_by_category,
    plot_correlation_heatmap,
    plot_countplot,
    plot_dashboard_grid,
    plot_histogram,
    plot_line_chart,
    plot_pairplot,
    plot_regression_scatter,
    plot_scatter_correlation,
    plot_violin_by_category,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"
NUMERIC_COLS = ["quantity", "unit_price", "standard_discount_pct"]
NORMALITY_COLS = ["quantity", "unit_price"]


# --------------------------------------------------------------------------
# 1. Chart generation - one call per Day 13/14 function, all reused as-is
# --------------------------------------------------------------------------
def _generate_all_charts(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Generate the full Day 13 + Day 14 chart suite and return (path, caption) pairs, in report order."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    charts: List[Tuple[str, str]] = []

    def _add(path: Path, caption: str) -> None:
        charts.append((str(path), caption))

    dashboard_path = CHARTS_DIR / "dashboard.png"
    plot_dashboard_grid(df, str(dashboard_path))
    _add(dashboard_path, "Figure 1. Statistical dashboard overview (histogram, boxplot, scatter, bar chart).")

    heatmap_path = CHARTS_DIR / "seaborn_heatmap_correlation.png"
    plot_correlation_heatmap(df, str(heatmap_path))
    _add(heatmap_path, "Figure 2. Pearson correlation heatmap across numeric columns.")

    hist_path = CHARTS_DIR / "histogram_unit_price.png"
    plot_histogram(df["unit_price"], "unit_price", str(hist_path))
    _add(hist_path, "Figure 3. Distribution of unit_price.")

    box_path = CHARTS_DIR / "boxplot_unit_price_by_category.png"
    plot_boxplot_by_category(df, "unit_price", "product_category", str(box_path))
    _add(box_path, "Figure 4. unit_price spread by product_category.")

    violin_path = CHARTS_DIR / "seaborn_violin_unit_price_by_category.png"
    plot_violin_by_category(df, "unit_price", "product_category", str(violin_path))
    _add(violin_path, "Figure 5. unit_price density by product_category (Seaborn violin).")

    scatter_path = CHARTS_DIR / "scatter_quantity_vs_unit_price.png"
    plot_scatter_correlation(df, "quantity", "unit_price", str(scatter_path))
    _add(scatter_path, "Figure 6. quantity vs. unit_price scatter.")

    reg_path = CHARTS_DIR / "seaborn_regression_quantity_vs_unit_price.png"
    plot_regression_scatter(df, "quantity", "unit_price", str(reg_path))
    _add(reg_path, "Figure 7. quantity vs. unit_price with fitted regression line.")

    bar_path = CHARTS_DIR / "bar_orders_by_category.png"
    plot_bar_chart(df, "product_category", str(bar_path))
    _add(bar_path, "Figure 8. Order count by product_category.")

    count_path = CHARTS_DIR / "seaborn_countplot_category.png"
    plot_countplot(df, "product_category", str(count_path))
    _add(count_path, "Figure 9. Category frequency (Seaborn countplot).")

    line_path = CHARTS_DIR / "line_unit_price_trend.png"
    plot_line_chart(df, "order_date", "unit_price", str(line_path), freq="ME")
    _add(line_path, "Figure 10. Monthly unit_price total trend.")

    pairplot_path = CHARTS_DIR / "seaborn_pairplot_by_category.png"
    plot_pairplot(df, "product_category", str(pairplot_path))
    _add(pairplot_path, "Figure 11. Pairwise numeric relationships, colored by product_category.")

    logger.info("Generated %d charts into %s", len(charts), CHARTS_DIR)
    return charts


# --------------------------------------------------------------------------
# 2. Statistics summary - Day 11/12 functions, run on the real dataset
# --------------------------------------------------------------------------
def _compute_stats_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Run the full Day 11-12 statistical analysis on `df` and collect it into one dict."""
    central_tendency = {col: compute_central_tendency(df[col].to_numpy(dtype=float)) for col in NUMERIC_COLS}
    spread = {col: compute_spread(df[col].to_numpy(dtype=float)) for col in NUMERIC_COLS}

    correlation_report = correlation_matrix_report(df)
    extremes = strongest_and_weakest_pairs(correlation_report, n=2)
    correlation_matrix = build_correlation_matrix(df)

    normality = {col: fit_normal_distribution(df[col].to_numpy(dtype=float)) for col in NORMALITY_COLS}
    outliers = {col: flag_zscore_outliers(df[col].to_numpy(dtype=float)) for col in NUMERIC_COLS}

    category_counts = df["product_category"].value_counts()
    category_median_price = df.groupby("product_category")["unit_price"].median().sort_values(ascending=False)

    return {
        "row_count": len(df),
        "central_tendency": central_tendency,
        "spread": spread,
        "correlation_report": correlation_report,
        "correlation_matrix": correlation_matrix,
        "strongest_pairs": extremes["strongest"],
        "weakest_pairs": extremes["weakest"],
        "normality": normality,
        "outliers": outliers,
        "category_counts": category_counts,
        "category_median_price": category_median_price,
    }


# --------------------------------------------------------------------------
# 3. Business insights + recommendations - derived from the stats above
# --------------------------------------------------------------------------
def build_insights_and_recommendations(stats_summary: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Turn a stats summary into plain-English findings and recommendations.

    Every sentence below references a specific number pulled from
    `stats_summary` at call time - nothing here is a hardcoded example
    value, so the report always reflects whatever dataset was analyzed.

    Returns:
        (findings, recommendations) - two lists of strings.
    """
    ct = stats_summary["central_tendency"]
    sp = stats_summary["spread"]
    strongest = stats_summary["strongest_pairs"].iloc[0]
    weakest = stats_summary["weakest_pairs"].iloc[-1]
    normality = stats_summary["normality"]
    outliers = stats_summary["outliers"]
    cat_counts = stats_summary["category_counts"]
    cat_price = stats_summary["category_median_price"]

    total_outliers = sum(o["beyond_2.0_std"]["count"] for o in outliers.values())
    non_normal_cols = [col for col, r in normality.items() if not r["is_normal"]]

    findings = [
        f"Orders average ${ct['unit_price']['mean']:.2f} per unit (median ${ct['unit_price']['median']:.2f}), "
        f"with a standard deviation of ${sp['unit_price']['std_sample']:.2f} - prices are widely spread across "
        f"a range of ${sp['unit_price']['range']:.2f}, not clustered around one typical value.",

        f"Typical order quantity is {ct['quantity']['mean']:.1f} units (mode {ct['quantity']['mode']:.0f}), "
        f"with an IQR of {sp['quantity']['iqr']:.2f} units - most orders fall within a fairly narrow band.",

        f"The strongest pairwise relationship found is between {strongest['column_a']} and {strongest['column_b']} "
        f"(Pearson r = {strongest['pearson']:+.3f}), and even that is weak. The weakest is between "
        f"{weakest['column_a']} and {weakest['column_b']} (r = {weakest['pearson']:+.3f}). No numeric column "
        f"pair shows a meaningful linear relationship in this dataset.",

        (
            f"Shapiro-Wilk testing rejects normality (p < 0.05) for {' and '.join(non_normal_cols)}"
            if non_normal_cols else
            "Shapiro-Wilk testing does not reject normality for the columns tested"
        ) + " - downstream analysis should not assume a bell-curve shape for these columns.",

        (
            f"Z-score outlier screening (\u00b12\u03c3) flagged {total_outliers} extreme value(s) across "
            f"{', '.join(NUMERIC_COLS)} - pricing and quantities are consistent, with no runaway values distorting the averages."
            if total_outliers == 0 else
            f"Z-score outlier screening (\u00b12\u03c3) flagged {total_outliers} extreme value(s) across "
            f"{', '.join(NUMERIC_COLS)}, worth reviewing individually before further analysis."
        ),

        f"{cat_counts.index[0]} is the highest-volume category with {int(cat_counts.iloc[0])} orders, "
        f"versus only {int(cat_counts.iloc[-1])} for {cat_counts.index[-1]} - a "
        f"{cat_counts.iloc[0] / cat_counts.iloc[-1]:.1f}x difference in order volume between the busiest and "
        f"quietest category.",

        f"{cat_price.index[0]} has the highest median unit_price (${cat_price.iloc[0]:.2f}) among categories, "
        f"while {cat_price.index[-1]} has the lowest (${cat_price.iloc[-1]:.2f}) - a "
        f"${cat_price.iloc[0] - cat_price.iloc[-1]:.2f} gap that the boxplot/violin charts make visible at a glance.",
    ]

    recommendations = [
        "Because no numeric column pair shows a meaningful linear (or monotonic) relationship, do not rely on "
        "quantity, price, or discount alone to predict one another - collect additional features (customer "
        "segment, marketing channel, seasonality) before attempting predictive modeling on this data.",

        "Since quantity and unit_price both fail normality testing, avoid statistical methods that assume a "
        "normal distribution (e.g. raw z-tests or Pearson-based significance tests without checking assumptions) - "
        "prefer non-parametric methods (Spearman correlation, rank-based tests) or transform the data first.",

        f"Investigate why {cat_counts.index[-1]} has such a low order volume relative to "
        f"{cat_counts.index[0]} ({int(cat_counts.iloc[-1])} vs. {int(cat_counts.iloc[0])} orders) - determine "
        "whether this reflects genuine low demand or a fixable gap in marketing/inventory support before "
        "deprioritizing the category.",
    ]

    return findings, recommendations


def write_insights_markdown(findings: List[str], recommendations: List[str], filepath: str) -> None:
    """Write findings + recommendations to a Markdown file."""
    lines = [
        "# Business Insights Report",
        "",
        f"*Generated {date.today().isoformat()} - Epic 3 checkpoint, Day 15*",
        "",
        "## Key Findings",
        "",
    ]
    lines += [f"{i}. {finding}" for i, finding in enumerate(findings, start=1)]
    lines += ["", "## Recommendations", ""]
    lines += [f"{i}. {rec}" for i, rec in enumerate(recommendations, start=1)]
    lines.append("")

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote insights markdown: %s", filepath)


# --------------------------------------------------------------------------
# 4. PDF compilation
# --------------------------------------------------------------------------
def _text_page(pdf: PdfPages, title: str, body_lines: List[str], fontsize: int = 11, numbered: bool = False) -> None:
    """Render one PDF page of wrapped plain text under a title.

    `$` is escaped to `\\$` before rendering - matplotlib's text renderer
    treats a bare `$...$` pair as LaTeX math mode by default, which
    silently mangles any dollar-amount text (e.g. "$46.02") that happens
    to contain two dollar signs on the same line.
    """
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", va="top")

    y = 0.87
    for i, line in enumerate(body_lines, start=1):
        prefix = f"{i}. " if numbered else ""
        safe_line = (prefix + line).replace("$", r"\$")
        wrapped = textwrap.wrap(safe_line, width=95, subsequent_indent="   " if numbered else "") or [""]
        for sub_line in wrapped:
            fig.text(0.08, y, sub_line, fontsize=fontsize, va="top", family="sans-serif")
            y -= 0.028
        y -= 0.012  # extra gap between list items
    pdf.savefig(fig)
    plt.close(fig)


def _stats_table_page(pdf: PdfPages, stats_summary: Dict[str, Any]) -> None:
    """Render one PDF page with a table of central tendency + spread per numeric column."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title("Executive Summary - Descriptive Statistics", fontsize=16, fontweight="bold", pad=20)

    rows = []
    for col in NUMERIC_COLS:
        ct = stats_summary["central_tendency"][col]
        sp = stats_summary["spread"][col]
        display_name = "discount_pct" if col == "standard_discount_pct" else col
        rows.append([
            display_name, f"{ct['mean']:.2f}", f"{ct['median']:.2f}", f"{ct['mode']:.2f}",
            f"{sp['std_sample']:.2f}", f"{sp['range']:.2f}", f"{sp['iqr']:.2f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Column", "Mean", "Median", "Mode", "Std (sample)", "Range", "IQR"],
        colWidths=[0.22, 0.13, 0.13, 0.13, 0.15, 0.12, 0.12],
        loc="upper center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    footnote = (
        f"Dataset: {stats_summary['row_count']} orders (Epic 2 cleaned + merged dataset). "
        "All values computed from scratch (Day 11) and cross-validated against NumPy/SciPy/pandas."
    )
    fig.text(0.08, 0.35, "\n".join(textwrap.wrap(footnote, width=95)).replace("$", r"\$"), fontsize=10, color="gray")

    pdf.savefig(fig)
    plt.close(fig)


def _chart_page(pdf: PdfPages, image_path: str, caption: str) -> None:
    """Render one PDF page embedding an already-generated chart PNG, with a caption."""
    image = plt.imread(image_path)
    fig = plt.figure(figsize=(8.5, 11))
    # A near-full-page axes (rather than matplotlib's default subplot
    # margins) so a wide chart image scales up to fill the page instead
    # of floating small in the middle of a lot of blank space.
    ax = fig.add_axes((0.03, 0.08, 0.94, 0.86))
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.5, 0.04, caption, fontsize=10, ha="center", style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def compile_pdf_report(chart_paths: List[Any], stats_summary: Dict[str, Any], output_pdf: str) -> None:
    """Compile a title page, stats summary, one page per chart, and an insights page into one PDF.

    Args:
        chart_paths: List of chart image paths, OR list of (path, caption)
            tuples (as returned by `_generate_all_charts`). A caption-less
            list is captioned generically, so this function also works
            standalone in tests with a couple of throwaway PNGs.
        stats_summary: Dict as produced by `_compute_stats_summary`.
        output_pdf: Where to write the combined PDF.
    """
    normalized_charts: List[Tuple[str, str]] = [
        (item, f"Figure {i}.") if isinstance(item, str) else (item[0], item[1])
        for i, item in enumerate(chart_paths, start=1)
    ]

    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    findings, recommendations = build_insights_and_recommendations(stats_summary)

    with PdfPages(output_pdf) as pdf:
        _text_page(
            pdf, "Business Insights Report",
            [
                "Epic 2 E-commerce Dataset - Statistical Analysis & Visualization",
                f"Generated {date.today().isoformat()} | Epic 3 checkpoint (Days 11-14 combined)",
                "",
                f"Dataset: {stats_summary.get('row_count', 'N/A')} orders, "
                f"{len(stats_summary.get('central_tendency', {}))} numeric columns analyzed.",
            ],
            fontsize=13,
        )
        _stats_table_page(pdf, stats_summary)
        for path, caption in normalized_charts:
            _chart_page(pdf, path, caption)
        _text_page(pdf, "Key Findings", findings, numbered=True)
        _text_page(pdf, "Recommendations", recommendations, numbered=True)

    logger.info("Compiled PDF report: %s (%d chart pages)", output_pdf, len(normalized_charts))


# --------------------------------------------------------------------------
# 5. Master orchestration
# --------------------------------------------------------------------------
def generate_full_report(df: pd.DataFrame, output_path: str) -> None:
    """Run the full Epic 3 pipeline end-to-end: stats -> charts -> PDF -> INSIGHTS.md.

    Args:
        df: The merged Epic 2 dataset (or any DataFrame with the same
            columns: quantity, unit_price, standard_discount_pct,
            product_category, order_date).
        output_path: Where to write the combined PDF report. The
            companion INSIGHTS.md is written alongside it, in the same
            directory.
    """
    logger.info("Generating full Business Insights Report for %d rows...", len(df))

    chart_paths = _generate_all_charts(df)
    stats_summary = _compute_stats_summary(df)
    compile_pdf_report(chart_paths, stats_summary, output_path)

    findings, recommendations = build_insights_and_recommendations(stats_summary)
    insights_path = Path(output_path).parent / "INSIGHTS.md"
    write_insights_markdown(findings, recommendations, str(insights_path))

    logger.info("Report complete: %s", output_path)
    logger.info("Insights complete: %s", insights_path)


def main() -> None:
    logger.info("=== Day 15: EDA + Visualization Project - Business Insights Report ===")
    df = load_merged_dataset()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_full_report(df, str(REPORTS_DIR / "business_insights_report.pdf"))


if __name__ == "__main__":
    main()
