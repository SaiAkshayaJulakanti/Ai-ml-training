"""Day 8: Filtering, Grouping & Merge/Join Operations

This module covers multi-condition filtering, groupby aggregation, and
combining datasets via merge/join/concat - built on top of Day 7's
cleaned canonical e-commerce orders dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CLEANED_CSV = DATA_DIR / "cleaned_dataset.csv"
CATEGORY_LOOKUP_CSV = DATA_DIR / "category_lookup.csv"


def _load_cleaned_orders() -> pd.DataFrame:
    """Load the Day 7 cleaned dataset with correct dtypes (CSV doesn't
    preserve dtype metadata, so order_date needs re-parsing on load -
    same fix as Day 6/7)."""
    df = pd.read_csv(CLEANED_CSV)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


# --------------------------------------------------------------------------
# 1. Multi-condition filtering
# --------------------------------------------------------------------------
def filtering_demo(df: pd.DataFrame) -> None:
    """Demonstrate multi-condition filtering with &, |, .isin(), .between(), .query()."""
    high_value_electronics = df[(df["product_category"] == "Electronics") & (df["unit_price"] > 200)]
    logger.info("AND filter (Electronics AND unit_price > 200): %d rows", len(high_value_electronics))

    cancelled_or_pending = df[(df["status"] == "Cancelled") | (df["status"] == "Pending")]
    logger.info("OR filter (Cancelled OR Pending): %d rows", len(cancelled_or_pending))

    selected_regions = df[df["region"].isin(["North", "South"])]
    logger.info(".isin() filter (region in [North, South]): %d rows", len(selected_regions))

    mid_price = df[df["unit_price"].between(50, 150)]
    logger.info(".between() filter (unit_price 50-150 inclusive): %d rows", len(mid_price))

    via_query = df.query("product_category == 'Sports' and quantity >= 5")
    logger.info(".query() filter (Sports AND quantity >= 5): %d rows", len(via_query))


def filter_records(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """Filter `df` using a dict of column -> condition.

    Each condition value can be:
        - a scalar: exact match (column == value)
        - a list/tuple: membership test (column.isin(value))
        - a 2-tuple explicitly meant as a range: use {'between': (lo, hi)}
          nested dict form for clarity - see the 'quantity' example below.

    Args:
        df: Input DataFrame.
        conditions: e.g. {"region": "North", "status": ["Delivered", "Shipped"],
                           "quantity": {"between": (2, 5)}}

    Returns:
        A new, filtered DataFrame (all conditions combined with AND).
    """
    mask = pd.Series(True, index=df.index)

    for col, condition in conditions.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        if isinstance(condition, dict) and "between" in condition:
            lo, hi = condition["between"]
            mask &= df[col].between(lo, hi)
        elif isinstance(condition, (list, tuple, set)):
            mask &= df[col].isin(condition)
        else:
            mask &= df[col] == condition

    return df[mask]


# --------------------------------------------------------------------------
# 2. Groupby aggregation
# --------------------------------------------------------------------------
def groupby_demo(df: pd.DataFrame) -> None:
    """Demonstrate groupby on single/multiple columns, .agg() with multiple
    functions, and .transform() for group-wise normalization."""
    single_col = df.groupby("product_category")["unit_price"].mean()
    logger.info("Single-column groupby (mean unit_price per category):\n%s", single_col)

    multi_col = df.groupby(["region", "status"])["quantity"].sum()
    logger.info("Multi-column groupby (sum quantity per region+status):\n%s", multi_col)

    multi_agg = df.groupby("product_category").agg(
        total_revenue=("unit_price", "sum"),
        avg_quantity=("quantity", "mean"),
        order_count=("order_id", "count"),
    )
    logger.info(".agg() with multiple named aggregations:\n%s", multi_agg)

    # .transform() returns a result the SAME SHAPE as the original DataFrame
    # (one value per row, not one per group) - unlike .agg()/.groupby().mean(),
    # which collapses down to one row per group. This is what makes
    # .transform() usable for group-wise normalization: each row gets its
    # OWN group's mean subtracted from it, without losing any rows.
    df = df.copy()
    df["category_avg_price"] = df.groupby("product_category")["unit_price"].transform("mean")
    df["price_vs_category_avg"] = df["unit_price"] - df["category_avg_price"]
    logger.info(
        ".transform() group-wise normalization (first 5 rows):\n%s",
        df[["product_category", "unit_price", "category_avg_price", "price_vs_category_avg"]].head(),
    )


def group_aggregate(df: pd.DataFrame, group_cols: List[str], agg_map: Dict[str, Any]) -> pd.DataFrame:
    """Group `df` by `group_cols` and aggregate according to `agg_map`.

    Args:
        df: Input DataFrame.
        group_cols: Column(s) to group by.
        agg_map: Passed directly to .agg() - maps column name to an
            aggregation function name (or list of names), e.g.
            {"unit_price": "sum", "quantity": ["mean", "max"]}.

    Returns:
        The aggregated DataFrame, with the group columns reset back into
        regular columns (not left as the index) for easier downstream use.
    """
    return df.groupby(group_cols).agg(agg_map).reset_index()


# --------------------------------------------------------------------------
# 3. Practical exercise: top_n_per_group
# --------------------------------------------------------------------------
def top_n_per_group(df: pd.DataFrame, group_col: str, sort_col: str, n: int) -> pd.DataFrame:
    """Return the top `n` rows per group in `group_col`, ranked by `sort_col` (descending).

    Args:
        df: Input DataFrame.
        group_col: Column defining the groups (e.g. 'customer_name').
        sort_col: Column to rank within each group (e.g. 'unit_price').
        n: Number of top rows to keep per group.

    Returns:
        A DataFrame containing only the top `n` rows of each group,
        sorted by group then by `sort_col` descending.
    """
    return (
        df.sort_values(sort_col, ascending=False)
        .groupby(group_col, group_keys=False)
        .head(n)
    )


# --------------------------------------------------------------------------
# 4. Merge / join operations
# --------------------------------------------------------------------------
def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, on: str, how: str) -> pd.DataFrame:
    """Merge `df1` and `df2` on column `on`, using join type `how`.

    Args:
        df1: Left DataFrame.
        df2: Right DataFrame.
        on: Column name to join on (must exist in both DataFrames).
        how: One of 'inner', 'left', 'right', 'outer'.

    Returns:
        The merged DataFrame.
    """
    return pd.merge(df1, df2, on=on, how=how)


def merge_join_demo(orders: pd.DataFrame, lookup: pd.DataFrame) -> None:
    """Demonstrate all 4 join types and their resulting row-count differences.

    WHEN TO USE MERGE vs JOIN vs CONCAT (documented here, per the task):
        - pd.merge(): combines two DataFrames SIDE BY SIDE based on
          matching key column(s) - like a SQL JOIN. Use this whenever
          you're combining related data that shares a common key
          (orders + a category lookup table, exactly as here).
        - DataFrame.join(): a convenience wrapper around merge() that
          joins on the INDEX by default rather than a column - useful
          when the key you want to join on is already the index of one
          or both DataFrames, saving a reset_index()/set_index() step.
        - pd.concat(): stacks DataFrames together - vertically (axis=0,
          the default: appending more ROWS, e.g. combining this month's
          and last month's orders which share the same columns) or
          horizontally (axis=1: appending more COLUMNS side by side,
          when two DataFrames share the same row index but hold
          different columns). concat does NOT match on a key column
          the way merge does - it just lines rows/columns up positionally
          (or by index, for axis=1).

    Row count intuition for the 4 merge types, given orders has 6
    categories all present in lookup, and lookup has 1 EXTRA category
    ("Garden & Outdoor") with zero matching orders:
        - inner: only rows with a match in BOTH sides -> exactly len(orders)
          rows here, since every order's category exists in lookup.
        - left: ALL rows from orders, matched lookup data where possible
          -> exactly len(orders) rows (same as inner here, since nothing
          in orders lacks a lookup match).
        - right: ALL rows from lookup, matched orders data where possible
          -> MORE rows than inner, because "Garden & Outdoor" appears
          once with NaN for every orders-side column (no matching orders).
        - outer: ALL rows from EITHER side -> same result as right here,
          since left already covers everything inner does.
    """
    inner = merge_datasets(orders, lookup, on="product_category", how="inner")
    left = merge_datasets(orders, lookup, on="product_category", how="left")
    right = merge_datasets(orders, lookup, on="product_category", how="right")
    outer = merge_datasets(orders, lookup, on="product_category", how="outer")

    logger.info(
        "Row counts - orders: %d, lookup: %d | inner: %d, left: %d, right: %d, outer: %d",
        len(orders), len(lookup), len(inner), len(left), len(right), len(outer),
    )

    unmatched_in_right = right[right["order_id"].isnull()]
    logger.info(
        "Lookup categories with NO matching orders (visible via right join): %s",
        unmatched_in_right["product_category"].tolist(),
    )


def concat_demo(orders: pd.DataFrame) -> None:
    """Demonstrate pd.concat() for vertical and horizontal stacking."""
    first_half = orders.iloc[: len(orders) // 2]
    second_half = orders.iloc[len(orders) // 2 :]
    vertical = pd.concat([first_half, second_half], ignore_index=True)
    logger.info(
        "Vertical concat: %d + %d rows -> %d rows (same columns, more rows)",
        len(first_half), len(second_half), len(vertical),
    )

    left_cols = orders[["order_id", "customer_name"]]
    right_cols = orders[["quantity", "unit_price"]]
    horizontal = pd.concat([left_cols, right_cols], axis=1)
    logger.info(
        "Horizontal concat: %d + %d columns -> %d columns (same rows, more columns)",
        left_cols.shape[1], right_cols.shape[1], horizontal.shape[1],
    )


# --------------------------------------------------------------------------
# 5. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 8: Filtering, Grouping & Merge/Join Operations ===")

    orders = _load_cleaned_orders()
    lookup = pd.read_csv(CATEGORY_LOOKUP_CSV)
    logger.info("Loaded orders: %s, lookup: %s", orders.shape, lookup.shape)

    logger.info("--- Filtering demo ---")
    filtering_demo(orders)

    logger.info("--- filter_records() practical exercise ---")
    filtered = filter_records(orders, {
        "region": "North",
        "status": ["Delivered", "Shipped"],
        "quantity": {"between": (2, 5)},
    })
    logger.info("filter_records result: %d rows", len(filtered))

    logger.info("--- Groupby demo ---")
    groupby_demo(orders)

    logger.info("--- group_aggregate() practical exercise ---")
    summary = group_aggregate(
        orders, ["product_category"],
        {"unit_price": ["sum", "mean"], "quantity": "sum", "order_id": "count"},
    )
    logger.info("Grouped summary report:\n%s", summary)

    logger.info("--- top_n_per_group() practical exercise (top 3 orders per customer) ---")
    top_orders = top_n_per_group(orders, group_col="customer_name", sort_col="unit_price", n=3)
    logger.info("top_n_per_group result: %d rows (from %d unique customers)",
                len(top_orders), orders["customer_name"].nunique())

    logger.info("--- Merge/join demo ---")
    merge_join_demo(orders, lookup)

    logger.info("--- Concat demo ---")
    concat_demo(orders)


if __name__ == "__main__":
    main()
