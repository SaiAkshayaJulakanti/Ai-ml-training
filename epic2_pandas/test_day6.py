"""Pytest suite for day6_series_dataframes.py.

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
    from epic2_pandas.day6_series_dataframes import (
        CANONICAL_CSV,
        CANONICAL_JSON,
        CANONICAL_XLSX,
        dataframe_summary,
        load_dataset,
    )
except ImportError:
    from day6_series_dataframes import (
        CANONICAL_CSV,
        CANONICAL_JSON,
        CANONICAL_XLSX,
        dataframe_summary,
        load_dataset,
    )


# --------------------------------------------------------------------------
# load_dataset - all 3 required formats
# --------------------------------------------------------------------------
def test_load_dataset_csv():
    df = load_dataset(str(CANONICAL_CSV))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 100
    assert df.shape[1] >= 5


def test_load_dataset_xlsx():
    df = load_dataset(str(CANONICAL_XLSX))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 100
    assert df.shape[1] >= 5


def test_load_dataset_json():
    df = load_dataset(str(CANONICAL_JSON))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 100
    assert df.shape[1] >= 5


def test_load_dataset_all_formats_have_same_shape():
    """All 3 formats represent the SAME dataset - shapes must match exactly."""
    df_csv = load_dataset(str(CANONICAL_CSV))
    df_xlsx = load_dataset(str(CANONICAL_XLSX))
    df_json = load_dataset(str(CANONICAL_JSON))
    assert df_csv.shape == df_xlsx.shape == df_json.shape


def test_load_dataset_all_formats_have_same_columns():
    df_csv = load_dataset(str(CANONICAL_CSV))
    df_xlsx = load_dataset(str(CANONICAL_XLSX))
    df_json = load_dataset(str(CANONICAL_JSON))
    assert list(df_csv.columns) == list(df_xlsx.columns) == list(df_json.columns)


# --------------------------------------------------------------------------
# load_dataset - error handling
# --------------------------------------------------------------------------
def test_load_dataset_missing_file_raises():
    """Edge case: a file that doesn't exist must raise FileNotFoundError, not crash unexpectedly."""
    with pytest.raises(FileNotFoundError):
        load_dataset("this_file_does_not_exist.csv")


def test_load_dataset_unsupported_extension_raises(tmp_path):
    """Edge case: an existing file with an unsupported extension must raise a clear ValueError."""
    unsupported = tmp_path / "data.txt"
    unsupported.write_text("some,csv,like,content\n1,2,3,4")
    with pytest.raises(ValueError):
        load_dataset(str(unsupported))


def test_load_dataset_malformed_file_raises(tmp_path):
    """Edge case: a malformed CSV-named file with invalid content should raise, not silently misparse."""
    malformed = tmp_path / "malformed.xlsx"
    malformed.write_text("this is not a real excel file")
    with pytest.raises(Exception):
        load_dataset(str(malformed))


# --------------------------------------------------------------------------
# dataframe_summary
# --------------------------------------------------------------------------
def test_dataframe_summary_shape():
    df = load_dataset(str(CANONICAL_CSV))
    summary = dataframe_summary(df)
    assert summary["shape"] == df.shape


def test_dataframe_summary_dtypes():
    df = load_dataset(str(CANONICAL_CSV))
    summary = dataframe_summary(df)
    assert set(summary["dtypes"].keys()) == set(df.columns)


def test_dataframe_summary_null_counts_accurate():
    """The canonical dataset has 5 intentionally-introduced nulls in customer_name."""
    df = load_dataset(str(CANONICAL_CSV))
    summary = dataframe_summary(df)
    assert summary["total_nulls"] == 5
    assert summary["null_counts"]["customer_name"] == 5


def test_dataframe_summary_on_dataframe_with_no_nulls():
    """Edge case: a DataFrame with zero nulls should report total_nulls == 0."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    summary = dataframe_summary(df)
    assert summary["total_nulls"] == 0


def test_dataframe_summary_memory_usage_is_positive():
    df = load_dataset(str(CANONICAL_CSV))
    summary = dataframe_summary(df)
    assert summary["memory_usage_bytes"] > 0


def test_load_dataset_order_date_is_genuine_datetime():
    """The dataset must have a real THIRD type (date), not just numeric + string."""
    for path in (CANONICAL_CSV, CANONICAL_XLSX, CANONICAL_JSON):
        df = load_dataset(str(path))
        assert pd.api.types.is_datetime64_any_dtype(df["order_date"]), (
            f"order_date in {path.name} is {df['order_date'].dtype}, expected a datetime64 dtype"
        )
