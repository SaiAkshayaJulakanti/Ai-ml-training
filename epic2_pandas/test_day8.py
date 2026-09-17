"""Pytest suite for day8_filter_group_merge.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic2_pandas.day8_filter_group_merge import (
        filter_records,
        group_aggregate,
        merge_datasets,
        top_n_per_group,
    )
except ImportError:
    from day8_filter_group_merge import (
        filter_records,
        group_aggregate,
        merge_datasets,
        top_n_per_group,
    )


# --------------------------------------------------------------------------
# A small, hand-computable dataset used across these tests.
# --------------------------------------------------------------------------
@pytest.fixture
def orders_df():
    return pd.DataFrame({
        "customer": ["Alice", "Alice", "Alice", "Bob", "Bob", "Carol"],
        "category": ["A", "B", "A", "A", "B", "A"],
        "region": ["North", "North", "South", "North", "South", "North"],
        "amount": [100.0, 200.0, 50.0, 300.0, 150.0, 400.0],
    })


@pytest.fixture
def lookup_df():
    return pd.DataFrame({
        "category": ["A", "B", "C"],   # 'C' has no matching orders
        "department": ["Dept1", "Dept2", "Dept3"],
    })


# --------------------------------------------------------------------------
# filter_records
# --------------------------------------------------------------------------
def test_filter_records_exact_match(orders_df):
    result = filter_records(orders_df, {"customer": "Alice"})
    assert len(result) == 3
    assert set(result["customer"]) == {"Alice"}


def test_filter_records_isin(orders_df):
    result = filter_records(orders_df, {"customer": ["Alice", "Bob"]})
    assert len(result) == 5


def test_filter_records_between(orders_df):
    result = filter_records(orders_df, {"amount": {"between": (100, 200)}})
    assert sorted(result["amount"].tolist()) == [100.0, 150.0, 200.0]


def test_filter_records_multiple_conditions_combined_with_and(orders_df):
    result = filter_records(orders_df, {"customer": "Alice", "category": "A"})
    assert len(result) == 2
    assert all(result["customer"] == "Alice")
    assert all(result["category"] == "A")


def test_filter_records_unknown_column_raises(orders_df):
    with pytest.raises(ValueError):
        filter_records(orders_df, {"nonexistent": "x"})


# --------------------------------------------------------------------------
# group_aggregate - checked against manually computed values
# --------------------------------------------------------------------------
def test_group_aggregate_sum_matches_manual_calculation(orders_df):
    # Category A amounts: 100 (Alice) + 50 (Alice) + 300 (Bob) + 400 (Carol) = 850
    # Category B amounts: 200 (Alice) + 150 (Bob) = 350
    result = group_aggregate(orders_df, ["category"], {"amount": "sum"})
    result_dict = dict(zip(result["category"], result["amount"]))
    assert result_dict["A"] == 850.0
    assert result_dict["B"] == 350.0


def test_group_aggregate_multiple_functions(orders_df):
    result = group_aggregate(orders_df, ["category"], {"amount": ["sum", "mean", "count"]})
    # MultiIndex columns from multiple agg functions - flatten for easy checking
    result.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in result.columns]
    category_a = result[result["category"] == "A"].iloc[0]
    assert category_a["amount_sum"] == 850.0
    assert category_a["amount_count"] == 4


def test_group_aggregate_multiple_group_columns(orders_df):
    result = group_aggregate(orders_df, ["category", "region"], {"amount": "sum"})
    assert len(result) == orders_df.groupby(["category", "region"]).ngroups


# --------------------------------------------------------------------------
# top_n_per_group
# --------------------------------------------------------------------------
def test_top_n_per_group_known_values(orders_df):
    """Alice has 3 orders (100, 200, 50) - top 2 by amount should be 200 and 100."""
    result = top_n_per_group(orders_df, group_col="customer", sort_col="amount", n=2)
    alice_rows = result[result["customer"] == "Alice"]
    assert sorted(alice_rows["amount"].tolist(), reverse=True) == [200.0, 100.0]


def test_top_n_per_group_fewer_rows_than_n(orders_df):
    """Edge case: a group with fewer rows than n should just return all of them."""
    result = top_n_per_group(orders_df, group_col="customer", sort_col="amount", n=10)
    carol_rows = result[result["customer"] == "Carol"]
    assert len(carol_rows) == 1  # Carol only has 1 order total


# --------------------------------------------------------------------------
# merge_datasets - all 4 join types, checked against expected row counts
# --------------------------------------------------------------------------
def test_merge_inner_join_row_count(orders_df, lookup_df):
    """Inner join: only categories present in BOTH - all 6 orders match (A and B both exist in lookup)."""
    result = merge_datasets(orders_df, lookup_df, on="category", how="inner")
    assert len(result) == len(orders_df)


def test_merge_left_join_row_count(orders_df, lookup_df):
    """Left join: ALL orders rows kept, regardless of lookup match."""
    result = merge_datasets(orders_df, lookup_df, on="category", how="left")
    assert len(result) == len(orders_df)


def test_merge_right_join_row_count(orders_df, lookup_df):
    """Right join: ALL lookup rows kept - category 'C' has no orders, adds exactly 1 extra row."""
    result = merge_datasets(orders_df, lookup_df, on="category", how="right")
    assert len(result) == len(orders_df) + 1


def test_merge_outer_join_row_count(orders_df, lookup_df):
    """Outer join: union of both sides - same as right here, since left already covers everything inner does."""
    result = merge_datasets(orders_df, lookup_df, on="category", how="outer")
    assert len(result) == len(orders_df) + 1


def test_merge_right_join_unmatched_category_has_null_orders_data(orders_df, lookup_df):
    result = merge_datasets(orders_df, lookup_df, on="category", how="right")
    unmatched = result[result["category"] == "C"]
    assert len(unmatched) == 1
    assert pd.isnull(unmatched.iloc[0]["customer"])
