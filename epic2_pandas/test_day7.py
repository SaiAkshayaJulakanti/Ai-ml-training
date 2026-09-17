"""Pytest suite for day7_data_cleaning.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic2_pandas.day7_data_cleaning import (
        CANONICAL_CSV,
        clean_dataset,
        data_quality_report,
        duplicate_report,
        fix_dtypes,
        handle_missing_values,
        inject_dirty_data,
        missing_value_report,
        remove_duplicates,
    )
except ImportError:
    from day7_data_cleaning import (
        CANONICAL_CSV,
        clean_dataset,
        data_quality_report,
        duplicate_report,
        fix_dtypes,
        handle_missing_values,
        inject_dirty_data,
        missing_value_report,
        remove_duplicates,
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", None, "Dave", "Alice"],
        "score": [85.0, np.nan, 70.0, 90.0, 85.0],
        "date_str": ["2026-01-01", "2026-01-02", "2026-01-03", None, "2026-01-01"],
    })


# --------------------------------------------------------------------------
# inject_dirty_data
# --------------------------------------------------------------------------
def test_inject_dirty_data_does_not_modify_original():
    original = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    original_copy = original.copy()
    inject_dirty_data(original, missing_pct=0.5, dup_count=2)
    pd.testing.assert_frame_equal(original, original_copy)


def test_inject_dirty_data_adds_nulls_and_duplicates():
    df = pd.DataFrame({"order_id": range(1, 21), "value": range(20)})
    dirty = inject_dirty_data(df, missing_pct=0.2, dup_count=3)
    assert dirty.isnull().sum().sum() > 0
    assert len(dirty) == len(df) + 3
    assert dirty["order_id"].isnull().sum() == 0  # ID column never nulled


# --------------------------------------------------------------------------
# missing_value_report
# --------------------------------------------------------------------------
def test_missing_value_report_on_empty_dataframe():
    """Edge case: an empty DataFrame must not crash the report."""
    empty = pd.DataFrame()
    report = missing_value_report(empty)
    assert "empty" in report.lower()


def test_missing_value_report_contains_column_names(sample_df):
    report = missing_value_report(sample_df)
    assert "name" in report
    assert "score" in report


# --------------------------------------------------------------------------
# handle_missing_values
# --------------------------------------------------------------------------
def test_handle_missing_values_mean_strategy(sample_df):
    result = handle_missing_values(sample_df, {"score": "mean"})
    assert result["score"].isnull().sum() == 0


def test_handle_missing_values_mode_strategy(sample_df):
    result = handle_missing_values(sample_df, {"name": "mode"})
    assert result["name"].isnull().sum() == 0


def test_handle_missing_values_mean_on_non_numeric_raises(sample_df):
    """Edge case: applying 'mean' to a text column must raise, not silently corrupt data."""
    with pytest.raises(ValueError):
        handle_missing_values(sample_df, {"name": "mean"})


def test_handle_missing_values_drop_strategy(sample_df):
    result = handle_missing_values(sample_df, {"name": "drop"})
    assert result["name"].isnull().sum() == 0
    assert len(result) < len(sample_df)


def test_handle_missing_values_column_fully_missing():
    """Edge case: a column that is 100% missing should still be fillable."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
    result = handle_missing_values(df, {"b": "mode"})
    # mode of an all-NaN column is empty, so nothing to fill with -
    # the column legitimately stays null, but this must not crash
    assert len(result) == 3


# --------------------------------------------------------------------------
# remove_duplicates
# --------------------------------------------------------------------------
def test_remove_duplicates_full_row(sample_df):
    result = remove_duplicates(sample_df)
    assert result.duplicated().sum() == 0


def test_remove_duplicates_subset_based():
    df = pd.DataFrame({"id": [1, 1, 2], "value": [100, 999, 200]})
    result = remove_duplicates(df, subset=["id"])
    assert len(result) == 2
    assert result.iloc[0]["value"] == 100  # first occurrence kept


# --------------------------------------------------------------------------
# fix_dtypes
# --------------------------------------------------------------------------
def test_fix_dtypes_datetime_conversion(sample_df):
    result = fix_dtypes(sample_df, {"date_str": "datetime"})
    assert pd.api.types.is_datetime64_any_dtype(result["date_str"])


def test_fix_dtypes_numeric_string_with_commas():
    """Edge case: a numeric column stored as strings with commas (e.g. '1,234')."""
    df = pd.DataFrame({"amount": ["1,234", "5,678", "100"]})
    result = fix_dtypes(df, {"amount": "float"})
    assert result["amount"].tolist() == [1234.0, 5678.0, 100.0]


def test_fix_dtypes_missing_column_raises(sample_df):
    with pytest.raises(ValueError):
        fix_dtypes(sample_df, {"nonexistent_column": "int"})


# --------------------------------------------------------------------------
# data_quality_report
# --------------------------------------------------------------------------
def test_data_quality_report_accurate_counts(sample_df):
    report = data_quality_report(sample_df)
    assert report["total_nulls"] == 3  # 1 in name, 1 in score, 1 in date_str
    assert report["duplicate_count"] >= 0


# --------------------------------------------------------------------------
# clean_dataset - the full pipeline
# --------------------------------------------------------------------------
def test_clean_dataset_zero_nulls_and_duplicates():
    clean_source = pd.read_csv(CANONICAL_CSV)
    dirty = inject_dirty_data(clean_source, missing_pct=0.1, dup_count=10)
    cleaned = clean_dataset(dirty)

    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.duplicated().sum() == 0


def test_clean_dataset_correct_dtypes():
    clean_source = pd.read_csv(CANONICAL_CSV)
    dirty = inject_dirty_data(clean_source, missing_pct=0.1, dup_count=10)
    cleaned = clean_dataset(dirty)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])
    assert pd.api.types.is_numeric_dtype(cleaned["unit_price"])


def test_clean_dataset_on_empty_dataframe():
    """Edge case: an empty DataFrame must pass through the pipeline without crashing."""
    empty = pd.DataFrame(columns=["order_id", "customer_name", "quantity", "unit_price", "order_date", "region", "status"])
    result = clean_dataset(empty)
    assert len(result) == 0
