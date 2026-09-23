"""Pytest suite for day13_matplotlib_charts.py.

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
    from epic3_stats_viz.day13_matplotlib_charts import (
        plot_bar_chart,
        plot_boxplot_by_category,
        plot_dashboard_grid,
        plot_histogram,
        plot_line_chart,
        plot_scatter_correlation,
    )
except ImportError:
    from day13_matplotlib_charts import (
        plot_bar_chart,
        plot_boxplot_by_category,
        plot_dashboard_grid,
        plot_histogram,
        plot_line_chart,
        plot_scatter_correlation,
    )


def _assert_valid_png(path: Path) -> None:
    assert path.exists(), f"expected a file at {path}"
    assert path.stat().st_size > 0, f"{path} was created but is empty"
    # A real PNG always starts with this 8-byte magic number.
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": range(1, 13),
        "quantity": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3],
        "unit_price": [10.0, 20.0, 15.0, 40.0, 55.0, 12.0, 22.0, 18.0, 42.0, 60.0, 25.0, 30.0],
        "product_category": ["A", "A", "A", "B", "B", "B", "C", "C", "A", "B", "C", "A"],
        "order_date": pd.date_range("2026-01-01", periods=12, freq="W"),
    })


# --------------------------------------------------------------------------
# plot_histogram
# --------------------------------------------------------------------------
def test_plot_histogram_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "hist.png"
    plot_histogram(sample_df["unit_price"], "unit_price", str(out))
    _assert_valid_png(out)


def test_plot_histogram_handles_empty_series_without_raising(tmp_path):
    out = tmp_path / "hist_empty.png"
    empty_series = pd.Series([], dtype=float)
    plot_histogram(empty_series, "unit_price", str(out))  # must not raise
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_boxplot_by_category
# --------------------------------------------------------------------------
def test_plot_boxplot_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "box.png"
    plot_boxplot_by_category(sample_df, "unit_price", "product_category", str(out))
    _assert_valid_png(out)


def test_plot_boxplot_handles_single_category_without_raising(tmp_path, sample_df):
    out = tmp_path / "box_single.png"
    single_cat_df = sample_df.copy()
    single_cat_df["product_category"] = "OnlyOne"
    plot_boxplot_by_category(single_cat_df, "unit_price", "product_category", str(out))  # must not raise
    _assert_valid_png(out)


def test_plot_boxplot_handles_empty_dataframe_without_raising(tmp_path, sample_df):
    out = tmp_path / "box_empty.png"
    empty_df = sample_df.iloc[0:0]
    plot_boxplot_by_category(empty_df, "unit_price", "product_category", str(out))  # must not raise
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_scatter_correlation
# --------------------------------------------------------------------------
def test_plot_scatter_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "scatter.png"
    plot_scatter_correlation(sample_df, "quantity", "unit_price", str(out))
    _assert_valid_png(out)


def test_plot_scatter_handles_empty_dataframe_without_raising(tmp_path, sample_df):
    out = tmp_path / "scatter_empty.png"
    empty_df = sample_df.iloc[0:0]
    plot_scatter_correlation(empty_df, "quantity", "unit_price", str(out))  # must not raise
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_bar_chart
# --------------------------------------------------------------------------
def test_plot_bar_chart_counts_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "bar_counts.png"
    plot_bar_chart(sample_df, "product_category", str(out))
    _assert_valid_png(out)


def test_plot_bar_chart_with_value_col_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "bar_values.png"
    plot_bar_chart(sample_df, "product_category", str(out), value_col="unit_price")
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_line_chart
# --------------------------------------------------------------------------
def test_plot_line_chart_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "line.png"
    plot_line_chart(sample_df, "order_date", "unit_price", str(out), freq="ME")
    _assert_valid_png(out)


def test_plot_line_chart_handles_empty_dataframe_without_raising(tmp_path, sample_df):
    out = tmp_path / "line_empty.png"
    empty_df = sample_df.iloc[0:0]
    plot_line_chart(empty_df, "order_date", "unit_price", str(out))  # must not raise
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_dashboard_grid
# --------------------------------------------------------------------------
def test_plot_dashboard_grid_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "dashboard.png"
    plot_dashboard_grid(sample_df, str(out))
    _assert_valid_png(out)


def test_plot_dashboard_grid_larger_than_single_chart(tmp_path, sample_df):
    # A 2x2 dashboard should produce a meaningfully larger file than a
    # single chart, as a loose sanity check that all 4 panels rendered.
    single_out = tmp_path / "single.png"
    dashboard_out = tmp_path / "dashboard2.png"
    plot_histogram(sample_df["unit_price"], "unit_price", str(single_out))
    plot_dashboard_grid(sample_df, str(dashboard_out))
    assert dashboard_out.stat().st_size > single_out.stat().st_size


# --------------------------------------------------------------------------
# save path handling
# --------------------------------------------------------------------------
def test_plot_creates_missing_parent_directories(tmp_path, sample_df):
    nested_out = tmp_path / "nested" / "sub" / "hist.png"
    plot_histogram(sample_df["unit_price"], "unit_price", str(nested_out))
    _assert_valid_png(nested_out)
