"""Day 7: Data Cleaning - Missing Values & Duplicates

This module covers systematic data cleaning: detecting and handling
missing values, duplicate rows, and inconsistent data types - built on
top of Day 6's canonical e-commerce orders dataset.
"""

from __future__ import annotations

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
CANONICAL_CSV = DATA_DIR / "ecommerce_orders.csv"
CLEANED_CSV = DATA_DIR / "cleaned_dataset.csv"


# --------------------------------------------------------------------------
# 1. Inject dirty data - simulate a real messy dataset
# --------------------------------------------------------------------------
def inject_dirty_data(df: pd.DataFrame, missing_pct: float = 0.1, dup_count: int = 10) -> pd.DataFrame:
    """Return a COPY of `df` with missing values and duplicate rows injected.

    Args:
        df: The clean source DataFrame (never modified in place).
        missing_pct: Fraction of cells (0.0-1.0) to randomly null out,
            spread across all columns except a designated ID column.
        dup_count: Number of duplicate rows to append.

    Returns:
        A new, "dirty" DataFrame - the original `df` is left untouched.
    """
    dirty = df.copy()
    rng = np.random.default_rng(seed=7)

    # Never null out the primary key - a missing ID would make a row
    # impossible to identify at all, which isn't a realistic "messy
    # data" scenario, just a broken one.
    columns_to_dirty = [c for c in dirty.columns if c != "order_id"]

    n_rows = len(dirty)
    for col in columns_to_dirty:
        n_missing = int(n_rows * missing_pct)
        if n_missing == 0:
            continue
        missing_idx = rng.choice(n_rows, size=n_missing, replace=False)
        dirty.loc[missing_idx, col] = np.nan

    if dup_count > 0:
        dup_rows = dirty.sample(n=min(dup_count, n_rows), random_state=7)
        dirty = pd.concat([dirty, dup_rows], ignore_index=True)

    return dirty


# --------------------------------------------------------------------------
# 2. Missing value detection
# --------------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> str:
    """Build a simple heatmap-style TEXT report of null distribution per column.

    Uses .isnull().sum() under the hood, then renders each column's null
    percentage as a row of '#' characters - a text-only stand-in for a
    visual heatmap, readable directly in a log or terminal.
    """
    total_rows = len(df)
    null_counts = df.isnull().sum()

    lines = ["Missing Value Report", "=" * 50]
    if total_rows == 0:
        lines.append("(empty DataFrame - no rows to report on)")
        return "\n".join(lines)

    for col, count in null_counts.items():
        pct = count / total_rows
        bar = "#" * int(pct * 40)
        lines.append(f"{col:<20} {count:>5} ({pct:>6.1%}) {bar}")
    lines.append("=" * 50)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 3. Missing value handling
# --------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """Handle missing values per column, according to `strategy`.

    Args:
        df: Input DataFrame (not modified in place - a copy is returned).
        strategy: Maps column name -> one of:
            'mean'   - fill with the column's mean (numeric columns only)
            'median' - fill with the column's median (numeric columns only)
            'mode'   - fill with the column's most frequent value
            'drop'   - drop rows where THIS column is null
            'ffill'  - forward-fill (carry the previous valid value forward)
            'bfill'  - backward-fill (carry the next valid value backward)

    Returns:
        A new DataFrame with missing values handled as specified.

    Raises:
        ValueError: If a strategy name isn't recognized, or 'mean'/'median'
            is requested for a non-numeric column.
    """
    result = df.copy()

    for col, method in strategy.items():
        if col not in result.columns:
            continue  # silently skip columns not present - keeps this reusable

        if method == "mean":
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(f"Cannot apply 'mean' to non-numeric column '{col}'")
            result[col] = result[col].fillna(result[col].mean())
        elif method == "median":
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(f"Cannot apply 'median' to non-numeric column '{col}'")
            result[col] = result[col].fillna(result[col].median())
        elif method == "mode":
            mode_values = result[col].mode()
            if len(mode_values) > 0:
                result[col] = result[col].fillna(mode_values.iloc[0])
        elif method == "drop":
            result = result.dropna(subset=[col])
        elif method == "ffill":
            result[col] = result[col].ffill()
        elif method == "bfill":
            result[col] = result[col].bfill()
        else:
            raise ValueError(
                f"Unknown strategy '{method}' for column '{col}'. "
                f"Expected one of: mean, median, mode, drop, ffill, bfill"
            )

    return result


# --------------------------------------------------------------------------
# 4. Duplicate detection and removal
# --------------------------------------------------------------------------
def duplicate_report(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """Report on duplicate rows in `df`, optionally considering only `subset` columns."""
    dup_mask = df.duplicated(subset=subset, keep="first")
    return {
        "total_rows": len(df),
        "duplicate_count": int(dup_mask.sum()),
        "duplicate_indices": df.index[dup_mask].tolist(),
    }


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows from `df`.

    Args:
        df: Input DataFrame.
        subset: If given, only these columns are considered when
            identifying duplicates (e.g. ['order_id'] to treat rows with
            the same order_id as duplicates, ignoring other column
            differences). If None, a full-row comparison is used - every
            column must match for two rows to count as duplicates.

    Returns:
        A new DataFrame with duplicates removed (first occurrence kept),
        and the index reset to be contiguous again.
    """
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. Dtype correction
# --------------------------------------------------------------------------
def fix_dtypes(df: pd.DataFrame, dtype_map: Dict[str, str]) -> pd.DataFrame:
    """Convert columns to the dtypes specified in `dtype_map`.

    Args:
        df: Input DataFrame.
        dtype_map: Maps column name -> target dtype. The special value
            'datetime' triggers pd.to_datetime() rather than a plain
            .astype(), since datetime parsing needs its own function
            (this mirrors the fix from Day 6's load_dataset).

    Returns:
        A new DataFrame with the requested columns converted.

    Raises:
        ValueError: If a requested column doesn't exist in `df`.
    """
    result = df.copy()

    for col, target_dtype in dtype_map.items():
        if col not in result.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        if target_dtype == "datetime":
            result[col] = pd.to_datetime(result[col], errors="coerce")
        elif target_dtype in ("int", "int64"):
            # Numeric strings sometimes carry stray whitespace or commas
            # (e.g. "1,234") - strip both before casting, or the cast
            # fails outright instead of just being slow. Checking for
            # dtype == object to decide whether to do this is NOT
            # reliable across pandas versions - pandas 3.x reports a
            # plain string column's dtype as 'str', not 'object' - so
            # this always runs the string-clean step on anything that
            # isn't already numeric, rather than gating on a specific
            # dtype name.
            if not pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].astype(str).str.replace(",", "", regex=False).str.strip()
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
        elif target_dtype in ("float", "float64"):
            if not pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].astype(str).str.replace(",", "", regex=False).str.strip()
            result[col] = pd.to_numeric(result[col], errors="coerce")
        else:
            result[col] = result[col].astype(target_dtype)

    return result


# --------------------------------------------------------------------------
# 6. Data quality report - before/after comparison
# --------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize a DataFrame's data quality: null counts, duplicate counts, dtypes.

    Designed to be called once BEFORE cleaning and once AFTER cleaning,
    so the two results can be compared directly to demonstrate real
    measurable improvement.
    """
    return {
        "shape": df.shape,
        "total_nulls": int(df.isnull().sum().sum()),
        "null_counts_per_column": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def print_before_after_comparison(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    """Log a readable before/after comparison of two data_quality_report() results."""
    logger.info("=" * 60)
    logger.info("DATA QUALITY: BEFORE vs AFTER CLEANING")
    logger.info("=" * 60)
    logger.info("%-25s %-15s %-15s", "Metric", "Before", "After")
    logger.info("-" * 60)
    logger.info("%-25s %-15s %-15s", "Shape", str(before["shape"]), str(after["shape"]))
    logger.info("%-25s %-15s %-15s", "Total nulls", before["total_nulls"], after["total_nulls"])
    logger.info("%-25s %-15s %-15s", "Duplicate rows", before["duplicate_count"], after["duplicate_count"])
    logger.info("=" * 60)


# --------------------------------------------------------------------------
# 7. Full cleaning pipeline
# --------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on `df`, in the correct order.

    Order matters here:
        1. Fix dtypes FIRST - so numeric/date columns are the right
           type before mean/median/datetime-based operations rely on it.
        2. Handle missing values - now that dtypes are correct, mean/
           median fills compute on genuinely numeric data.
        3. Remove duplicates LAST - so that rows which only became
           identical to each other AFTER missing-value filling are
           also correctly caught (filling can make two previously-
           distinct rows collapse into duplicates of each other).

    Args:
        df: The raw (possibly dirty) input DataFrame.

    Returns:
        A fully cleaned DataFrame: zero nulls, zero duplicates, correct dtypes.
    """
    dtype_map = {
        "order_id": "int",
        "quantity": "int",
        "unit_price": "float",
        "order_date": "datetime",
    }
    result = fix_dtypes(df, {k: v for k, v in dtype_map.items() if k in df.columns})

    strategy = {
        "customer_name": "mode",
        "product_category": "mode",
        "quantity": "median",
        "unit_price": "median",
        "order_date": "ffill",
        "region": "mode",
        "status": "mode",
    }
    result = handle_missing_values(result, {k: v for k, v in strategy.items() if k in result.columns})

    result = remove_duplicates(result)

    return result


# --------------------------------------------------------------------------
# 8. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 7: Data Cleaning - Missing Values & Duplicates ===")

    logger.info("--- Loading canonical dataset and injecting dirty data ---")
    clean_source = pd.read_csv(CANONICAL_CSV)
    dirty = inject_dirty_data(clean_source, missing_pct=0.1, dup_count=10)
    logger.info("Dirty dataset shape: %s (source was %s)", dirty.shape, clean_source.shape)

    logger.info("--- Missing value report (before cleaning) ---")
    logger.info("\n%s", missing_value_report(dirty))

    logger.info("--- Duplicate report (before cleaning) ---")
    dup_before = duplicate_report(dirty)
    logger.info("Duplicates found: %d", dup_before["duplicate_count"])

    before_quality = data_quality_report(dirty)

    logger.info("--- Running full cleaning pipeline ---")
    cleaned = clean_dataset(dirty)

    after_quality = data_quality_report(cleaned)

    logger.info("--- Missing value report (after cleaning) ---")
    logger.info("\n%s", missing_value_report(cleaned))

    print_before_after_comparison(before_quality, after_quality)

    logger.info("--- Saving cleaned dataset ---")
    cleaned.to_csv(CLEANED_CSV, index=False)
    logger.info("Cleaned dataset saved to: %s", CLEANED_CSV)


if __name__ == "__main__":
    main()
