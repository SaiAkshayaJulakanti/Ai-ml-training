"""Day 9: Exploratory Data Analysis (EDA) - Epic 2 Checkpoint Deliverable

This module runs a structured exploratory analysis on the merged/cleaned
dataset from Days 7-8: univariate statistics, categorical value counts,
correlation, IQR-based outlier detection, pivot tables, and bin-based
categorization - combined into one full EDA report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"
CLEANED_CSV = DATA_DIR / "cleaned_dataset.csv"
CATEGORY_LOOKUP_CSV = DATA_DIR / "category_lookup.csv"
EDA_REPORT_JSON = REPORTS_DIR / "eda_report.json"
INSIGHTS_MD = REPORTS_DIR / "INSIGHTS.md"


def _load_merged_dataset() -> pd.DataFrame:
    """Load and merge the Day 7 cleaned orders with the Day 8 category lookup.

    This is the "merged dataset from Day 8" the task refers to - built
    fresh here via the same left join demonstrated on Day 8, rather than
    duplicating a saved copy of it.
    """
    orders = pd.read_csv(CLEANED_CSV)
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    lookup = pd.read_csv(CATEGORY_LOOKUP_CSV)
    return pd.merge(orders, lookup, on="product_category", how="left")


def _to_native(obj: Any) -> Any:
    """Recursively convert NumPy/pandas types to plain Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return _to_native(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _to_native(obj.to_dict())
    return obj


# --------------------------------------------------------------------------
# 1. Univariate statistics + categorical value counts + correlation
# --------------------------------------------------------------------------
def univariate_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute univariate statistics for numeric columns, value counts for
    categorical columns, and the correlation matrix between numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        A dict with keys:
            - 'numeric_stats': per numeric column - mean/median/std/min/max/25%/75%
            - 'categorical_value_counts': per non-numeric column - value -> count
            - 'correlation_matrix': numeric-column-to-numeric-column correlation
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Excluding datetime columns by passing a dtype STRING to select_dtypes
    # is fragile - pandas can reject an overly-specific string like
    # "datetime64[us]" outright (a real error hit while building this:
    # "'datetime64[us]' is too specific of a frequency"). Checking each
    # column directly with is_datetime64_any_dtype sidesteps needing to
    # guess the exact dtype string pandas wants.
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    numeric_stats = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "25%": float(df[col].quantile(0.25)),
            "75%": float(df[col].quantile(0.75)),
        }

    categorical_value_counts = {
        col: df[col].value_counts().to_dict() for col in categorical_cols
    }

    correlation_matrix = df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {}

    return {
        "numeric_stats": numeric_stats,
        "categorical_value_counts": categorical_value_counts,
        "correlation_matrix": correlation_matrix,
    }


# --------------------------------------------------------------------------
# 2. Outlier detection via IQR
# --------------------------------------------------------------------------
def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Flag outliers in `column` using the IQR method.

    A value is an outlier if it falls below Q1 - 1.5*IQR or above
    Q3 + 1.5*IQR, where IQR = Q3 - Q1. This is a standard, distribution-
    free rule of thumb (doesn't assume normality, unlike a z-score cutoff).

    Args:
        df: Input DataFrame.
        column: The numeric column to check.

    Returns:
        A DataFrame containing only the rows flagged as outliers in `column`.

    Raises:
        ValueError: If `column` doesn't exist or isn't numeric.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric - IQR outlier detection requires numeric data")

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


# --------------------------------------------------------------------------
# 3. Pivot tables
# --------------------------------------------------------------------------
def build_pivot_summary(df: pd.DataFrame, index: str, columns: str, values: str, aggfunc: str = "mean") -> pd.DataFrame:
    """Build a pivot table cross-tabulating `index` against `columns`, summarizing `values`.

    Args:
        df: Input DataFrame.
        index: Column to use as pivot table rows.
        columns: Column to use as pivot table columns.
        values: Column to aggregate.
        aggfunc: Aggregation function name (e.g. 'mean', 'sum', 'count').

    Returns:
        The pivot table as a DataFrame.
    """
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)


# --------------------------------------------------------------------------
# 4. Custom transformations - .apply(), .map(), pd.cut()
# --------------------------------------------------------------------------
def categorize_column(df: pd.DataFrame, column: str, bins: List[float], labels: List[str]) -> pd.DataFrame:
    """Bin a numeric column into labeled categories using pd.cut.

    Args:
        df: Input DataFrame (not modified in place).
        column: The numeric column to bin.
        bins: Bin edges, e.g. [0, 50, 150, 500] for 2 bins: (0,50], (50,150], (150,500].
              len(bins) must be len(labels) + 1.
        labels: Label for each bin, e.g. ["Low", "Medium", "High"].

    Returns:
        A new DataFrame with an added column named f"{column}_category".
    """
    result = df.copy()
    result[f"{column}_category"] = pd.cut(result[column], bins=bins, labels=labels)
    return result


def transformation_demo(df: pd.DataFrame) -> None:
    """Demonstrate .apply() and .map() for custom row/column-wise transformations."""
    # .map() - element-wise, on a SINGLE Series. Good for simple value
    # substitution or a one-argument function applied independently to
    # each value.
    status_short = df["status"].map({
        "Delivered": "OK", "Shipped": "OK", "Pending": "WAIT", "Cancelled": "CXL",
    })
    logger.info(".map() result (status shortened):\n%s", status_short.head())

    # .apply() on a Series - same idea as .map() but works with any
    # callable, not just a dict/simple function - useful when the logic
    # needs more than a single-expression lookup.
    price_tier = df["unit_price"].apply(lambda x: "premium" if x > 300 else "standard")
    logger.info(".apply() on a Series (price tier):\n%s", price_tier.head())

    # .apply() on a DataFrame with axis=1 - ROW-WISE, receives an entire
    # row at a time, letting the logic combine MULTIPLE columns together
    # (impossible with .map(), which only ever sees one column's values).
    order_value = df.apply(lambda row: row["quantity"] * row["unit_price"], axis=1)
    logger.info(".apply(axis=1) row-wise (order_value = quantity * unit_price):\n%s", order_value.head())


# --------------------------------------------------------------------------
# 5. Full EDA report - orchestrates everything
# --------------------------------------------------------------------------
def generate_eda_report(
    df: pd.DataFrame,
    pivot_specs: Optional[List[Dict[str, str]]] = None,
    categorize_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate the full EDA pipeline into one structured report dict.

    Genuinely generic: works on any DataFrame, not just the canonical
    e-commerce dataset - auto-selects columns when not told which to use.
    An earlier version hardcoded 'unit_price'/'product_category'/etc.
    directly, which crashed with a KeyError on any other DataFrame
    despite the function's signature promising to accept "a DataFrame",
    not specifically the canonical dataset.

    Args:
        df: The dataset to analyze (any DataFrame with at least one
            numeric column; pivot tables and categorization are skipped
            gracefully if there aren't enough numeric/categorical
            columns to build them from).
        pivot_specs: Optional list of up to 2 dicts, each with keys
            'index', 'columns', 'values', 'aggfunc' - lets the caller
            request specific, meaningful cross-tabulations (e.g. category
            x region) instead of relying on auto-detected column order,
            which tends to pick whichever categorical column happens to
            come first rather than the most analytically useful one.
        categorize_col: Optional column name to bin into Low/Medium/High
            categories. Defaults to the first numeric column if omitted.

    Returns:
        A dict combining univariate stats, outlier findings, pivot table
        summaries (where possible), and a categorized column sample.
    """
    report: Dict[str, Any] = {"shape": df.shape}

    univariate = univariate_report(df)
    report["univariate"] = univariate

    numeric_cols = list(univariate["numeric_stats"].keys())
    categorical_cols = list(univariate["categorical_value_counts"].keys())

    outliers = {}
    for col in numeric_cols:
        outlier_rows = detect_outliers_iqr(df, col)
        outliers[col] = {
            "count": len(outlier_rows),
            "outlier_values": outlier_rows[col].tolist(),
        }
    report["outliers"] = outliers

    # Pivot tables need at least 2 categorical columns (index + columns)
    # and 1 numeric column (values). Use caller-provided specs when
    # given (for a meaningful, business-relevant demo); otherwise fall
    # back to auto-detecting from whatever columns are available, so the
    # function never crashes on a narrower or differently-shaped dataset.
    report["pivot_tables"] = {}
    if pivot_specs:
        for spec in pivot_specs:
            name = f"{spec['index']}_by_{spec['columns']}_{spec.get('aggfunc', 'mean')}_{spec['values']}"
            report["pivot_tables"][name] = build_pivot_summary(
                df, index=spec["index"], columns=spec["columns"],
                values=spec["values"], aggfunc=spec.get("aggfunc", "mean"),
            )
    elif len(categorical_cols) >= 2 and numeric_cols:
        idx_col, col_col = categorical_cols[0], categorical_cols[1]
        value_col = numeric_cols[0]
        report["pivot_tables"][f"{idx_col}_by_{col_col}_avg_{value_col}"] = build_pivot_summary(
            df, index=idx_col, columns=col_col, values=value_col, aggfunc="mean"
        )
        report["pivot_tables"][f"{col_col}_by_{idx_col}_count"] = build_pivot_summary(
            df, index=col_col, columns=idx_col, values=value_col, aggfunc="count"
        )

    # Categorize a numeric column using ITS OWN quartiles as bin edges -
    # this is what makes the binning generic across any numeric column/
    # dataset, rather than assuming a specific value range (like a
    # hardcoded 0-100-300 price scale that would misbehave badly on a
    # column with a totally different scale, e.g. ages or counts).
    target_col = categorize_col if categorize_col else (numeric_cols[0] if numeric_cols else None)
    if target_col:
        col_min = df[target_col].min()
        col_max = df[target_col].max()
        q1 = df[target_col].quantile(0.33)
        q2 = df[target_col].quantile(0.66)
        bins = sorted(set([col_min - 1, q1, q2, col_max + 1]))
        labels = ["Low", "Medium", "High"][: len(bins) - 1]
        categorized = categorize_column(df, target_col, bins=bins, labels=labels)
        report["categorized_column"] = target_col
        report["category_counts"] = categorized[f"{target_col}_category"].value_counts().to_dict()

    return report


def save_eda_report(report: Dict[str, Any], filepath: Path) -> None:
    """Save the EDA report dict to a JSON file, converting all NumPy/pandas types first."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_to_native(report), f, indent=2)


# --------------------------------------------------------------------------
# 6. Insights - written in plain English, from the REAL computed numbers
# --------------------------------------------------------------------------
def generate_insights(report: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    """Derive plain-English insight bullet points from the actual computed report.

    Every insight here is computed from real numbers in `report`/`df` -
    none of the specific figures are hardcoded, so these regenerate
    correctly if the underlying dataset changes. Column names ARE
    specific to the canonical e-commerce dataset here (unlike
    generate_eda_report, which is generic) - insights are inherently
    domain-specific commentary, so this function is written for the
    actual dataset this training week's report is about, checking for
    each expected column before using it so it degrades gracefully
    rather than crashing on a differently-shaped DataFrame.
    """
    insights: List[str] = []
    numeric_stats = report["univariate"]["numeric_stats"]

    if "unit_price" in numeric_stats:
        price_stats = numeric_stats["unit_price"]
        is_right_skewed = price_stats["mean"] > price_stats["median"]
        insights.append(
            f"Average order unit price is ${price_stats['mean']:.2f}, with a "
            f"median of ${price_stats['median']:.2f} - the mean sitting "
            f"{'above' if is_right_skewed else 'below'} "
            f"the median suggests a {'right' if is_right_skewed else 'left'}-skewed "
            f"price distribution (a few {'high' if is_right_skewed else 'low'}-price orders "
            f"pulling the average {'up' if is_right_skewed else 'down'})."
        )

    if "product_category" in df.columns and "unit_price" in df.columns:
        category_means = df.groupby("product_category")["unit_price"].mean()
        top_category = category_means.idxmax()
        bottom_category = category_means.idxmin()
        pct_diff = (category_means[top_category] / category_means[bottom_category] - 1) * 100
        insights.append(
            f"'{top_category}' has the highest average order value (${category_means[top_category]:.2f}), "
            f"{pct_diff:.0f}% higher than the lowest category '{bottom_category}' "
            f"(${category_means[bottom_category]:.2f})."
        )

    total_outliers = sum(v["count"] for v in report["outliers"].values())
    outlier_cols = [col for col, v in report["outliers"].items() if v["count"] > 0]
    if outlier_cols:
        insights.append(
            f"{total_outliers} outlier value(s) detected via the IQR method across "
            f"{len(outlier_cols)} column(s) ({', '.join(outlier_cols)}) - worth reviewing "
            f"before these values feed into any downstream averages or models."
        )
    else:
        insights.append("No outliers were detected via the IQR method in any numeric column.")

    if "status" in df.columns:
        status_counts = df["status"].value_counts()
        delivered_pct = status_counts.get("Delivered", 0) / len(df) * 100
        cancelled_pct = status_counts.get("Cancelled", 0) / len(df) * 100
        insights.append(
            f"{delivered_pct:.1f}% of orders are marked Delivered, while "
            f"{cancelled_pct:.1f}% are Cancelled."
        )

    if "category_counts" in report:
        category_counts = report["category_counts"]
        most_common_tier = max(category_counts, key=category_counts.get)
        insights.append(
            f"For '{report['categorized_column']}', the '{most_common_tier}' tier is the "
            f"most common, covering {category_counts[most_common_tier]} of {len(df)} rows "
            f"({category_counts[most_common_tier] / len(df) * 100:.1f}%)."
        )

    return insights


def save_insights(insights: List[str], filepath: Path) -> None:
    """Save the insights list as a simple Markdown bullet list."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# EDA Insights", ""]
    lines.extend(f"- {insight}" for insight in insights)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# 7. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 9: Exploratory Data Analysis (Epic 2 Checkpoint) ===")

    df = _load_merged_dataset()
    logger.info("Loaded merged dataset: %s", df.shape)

    logger.info("--- Transformation demo (.apply, .map) ---")
    transformation_demo(df)

    logger.info("--- Generating full EDA report ---")
    report = generate_eda_report(
        df,
        pivot_specs=[
            {"index": "product_category", "columns": "region", "values": "unit_price", "aggfunc": "mean"},
            {"index": "status", "columns": "department", "values": "order_id", "aggfunc": "count"},
        ],
        categorize_col="unit_price",
    )

    logger.info("Numeric stats (unit_price): %s", report["univariate"]["numeric_stats"]["unit_price"])
    logger.info("Outliers found: %s", {k: v["count"] for k, v in report["outliers"].items()})
    for pivot_name, pivot_table in report["pivot_tables"].items():
        logger.info("Pivot table '%s':\n%s", pivot_name, pivot_table)
    logger.info("Category counts (%s): %s", report.get("categorized_column"), report.get("category_counts"))

    save_eda_report(report, EDA_REPORT_JSON)
    logger.info("EDA report saved to: %s", EDA_REPORT_JSON)

    logger.info("--- Generating insights ---")
    insights = generate_insights(report, df)
    for i, insight in enumerate(insights, 1):
        logger.info("%d. %s", i, insight)

    save_insights(insights, INSIGHTS_MD)
    logger.info("Insights saved to: %s", INSIGHTS_MD)


if __name__ == "__main__":
    main()
