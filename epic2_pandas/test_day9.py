"""Pytest suite for day9_eda.py.

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
    from epic2_pandas.day9_eda import (
        build_pivot_summary,
        categorize_column,
        detect_outliers_iqr,
        generate_eda_report,
        generate_insights,
        univariate_report,
    )
except ImportError:
    from day9_eda import (
        build_pivot_summary,
        categorize_column,
        detect_outliers_iqr,
        generate_eda_report,
        generate_insights,
        univariate_report,
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "category": ["A", "A", "B", "B", "A", "B"],
        "region": ["North", "South", "North", "South", "North", "North"],
        "value": [10, 12, 11, 13, 12, 200],   # 200 is a deliberate outlier
        "status": ["Delivered", "Delivered", "Pending", "Delivered", "Cancelled", "Delivered"],
    })


# --------------------------------------------------------------------------
# univariate_report
# --------------------------------------------------------------------------
def test_univariate_report_numeric_stats(sample_df):
    report = univariate_report(sample_df)
    assert "value" in report["numeric_stats"]
    assert report["numeric_stats"]["value"]["min"] == 10
    assert report["numeric_stats"]["value"]["max"] == 200


def test_univariate_report_categorical_value_counts(sample_df):
    report = univariate_report(sample_df)
    assert report["categorical_value_counts"]["status"]["Delivered"] == 4


def test_univariate_report_correlation_matrix_shape(sample_df):
    df = sample_df.copy()
    df["value2"] = df["value"] * 2
    report = univariate_report(df)
    assert "value" in report["correlation_matrix"]
    assert "value2" in report["correlation_matrix"]["value"]


# --------------------------------------------------------------------------
# detect_outliers_iqr - against a manually constructed dataset with known outliers
# --------------------------------------------------------------------------
def test_detect_outliers_iqr_known_outliers():
    df = pd.DataFrame({"value": [10, 12, 11, 13, 12, 11, 10, 12, 100, -50]})
    # Q1=11, Q3=12, IQR=1 -> bounds: [9.5, 13.5] -> 100 and -50 are outliers
    result = detect_outliers_iqr(df, "value")
    assert set(result["value"].tolist()) == {100, -50}
    assert len(result) == 2


def test_detect_outliers_iqr_no_outliers():
    """Edge case: a tight, uniform dataset should flag nothing."""
    df = pd.DataFrame({"value": [10, 10, 11, 10, 11, 10]})
    result = detect_outliers_iqr(df, "value")
    assert len(result) == 0


def test_detect_outliers_iqr_non_numeric_column_raises():
    df = pd.DataFrame({"name": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        detect_outliers_iqr(df, "name")


def test_detect_outliers_iqr_missing_column_raises(sample_df):
    with pytest.raises(ValueError):
        detect_outliers_iqr(sample_df, "nonexistent")


# --------------------------------------------------------------------------
# build_pivot_summary - shape and values
# --------------------------------------------------------------------------
def test_build_pivot_summary_shape(sample_df):
    pivot = build_pivot_summary(sample_df, index="category", columns="region", values="value", aggfunc="mean")
    assert set(pivot.index) == {"A", "B"}
    assert set(pivot.columns) == {"North", "South"}


def test_build_pivot_summary_values_match_manual_calculation(sample_df):
    """Category A, region North: values 10 and 12 -> mean 11."""
    pivot = build_pivot_summary(sample_df, index="category", columns="region", values="value", aggfunc="mean")
    assert pivot.loc["A", "North"] == 11.0


# --------------------------------------------------------------------------
# categorize_column - bin assignment correctness
# --------------------------------------------------------------------------
def test_categorize_column_bin_assignment(sample_df):
    result = categorize_column(sample_df, "value", bins=[0, 12, 20, 300], labels=["Low", "Medium", "High"])
    # value=10 -> Low, value=13 -> Medium, value=200 -> High
    assert result.loc[0, "value_category"] == "Low"
    assert result.loc[3, "value_category"] == "Medium"
    assert result.loc[5, "value_category"] == "High"


def test_categorize_column_does_not_modify_original(sample_df):
    original_columns = list(sample_df.columns)
    categorize_column(sample_df, "value", bins=[0, 50, 300], labels=["Low", "High"])
    assert list(sample_df.columns) == original_columns


# --------------------------------------------------------------------------
# generate_eda_report - the full orchestration
# --------------------------------------------------------------------------
def test_generate_eda_report_has_all_expected_sections(sample_df):
    report = generate_eda_report(sample_df)
    assert "univariate" in report
    assert "outliers" in report
    assert "shape" in report


def test_generate_eda_report_works_on_generic_dataframe():
    """The function's signature promises to accept ANY DataFrame - verify it
    doesn't crash on data shaped nothing like the canonical dataset."""
    generic_df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": ["p", "q", "p", "q", "p"],
        "z": ["m", "m", "n", "n", "m"],
    })
    report = generate_eda_report(generic_df)
    assert report["shape"] == (5, 3)
    assert "x" in report["univariate"]["numeric_stats"]


def test_generate_eda_report_respects_explicit_pivot_specs(sample_df):
    report = generate_eda_report(
        sample_df,
        pivot_specs=[{"index": "category", "columns": "region", "values": "value", "aggfunc": "mean"}],
    )
    assert "category_by_region_mean_value" in report["pivot_tables"]


def test_generate_eda_report_outliers_include_known_outlier(sample_df):
    report = generate_eda_report(sample_df)
    assert report["outliers"]["value"]["count"] >= 1
    assert 200 in report["outliers"]["value"]["outlier_values"]


# --------------------------------------------------------------------------
# generate_insights
# --------------------------------------------------------------------------
def test_generate_insights_returns_multiple_strings(sample_df):
    report = generate_eda_report(sample_df)
    insights = generate_insights(report, sample_df)
    assert len(insights) >= 3
    assert all(isinstance(i, str) and len(i) > 0 for i in insights)
