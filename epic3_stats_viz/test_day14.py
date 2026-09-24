"""Pytest suite for day14_seaborn_advanced.py.

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
    from epic3_stats_viz.day14_seaborn_advanced import (
        build_correlation_matrix,
        plot_correlation_heatmap,
        plot_countplot,
        plot_matplotlib_vs_seaborn_comparison,
        plot_pairplot,
        plot_regression_scatter,
        plot_violin_by_category,
    )
except ImportError:
    from day14_seaborn_advanced import (
        build_correlation_matrix,
        plot_correlation_heatmap,
        plot_countplot,
        plot_matplotlib_vs_seaborn_comparison,
        plot_pairplot,
        plot_regression_scatter,
        plot_violin_by_category,
    )

TOLERANCE = 1e-6


def _assert_valid_png(path: Path) -> None:
    assert path.exists(), f"expected a file at {path}"
    assert path.stat().st_size > 0, f"{path} was created but is empty"
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 40
    return pd.DataFrame({
        "order_id": range(1, n + 1),
        "quantity": rng.integers(1, 10, size=n),
        "unit_price": rng.normal(200, 50, size=n),
        "standard_discount_pct": rng.uniform(0, 15, size=n),
        "product_category": rng.choice(["A", "B", "C"], size=n),
    })


@pytest.fixture
def known_correlation_df() -> pd.DataFrame:
    # a and b are a PERFECT positive line (r=1.0 exactly); c is designed
    # to be a perfect negative line against a (r=-1.0 exactly) - both
    # values are known ahead of time, independent of build_correlation_matrix.
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return pd.DataFrame({
        "a": a,
        "b": 3 * a + 7,
        "c": -2 * a + 1,
    })


# --------------------------------------------------------------------------
# build_correlation_matrix / plot_correlation_heatmap
# --------------------------------------------------------------------------
def test_build_correlation_matrix_matches_known_values(known_correlation_df):
    matrix = build_correlation_matrix(known_correlation_df)
    assert matrix.loc["a", "a"] == pytest.approx(1.0, abs=TOLERANCE)
    assert matrix.loc["a", "b"] == pytest.approx(1.0, abs=TOLERANCE)     # perfect positive
    assert matrix.loc["a", "c"] == pytest.approx(-1.0, abs=TOLERANCE)    # perfect negative
    assert matrix.loc["b", "c"] == pytest.approx(-1.0, abs=TOLERANCE)    # b and c both linear in a, opposite sign


def test_build_correlation_matrix_is_symmetric(sample_df):
    matrix = build_correlation_matrix(sample_df)
    numeric_matrix = matrix.astype(float)
    assert np.allclose(numeric_matrix.to_numpy(), numeric_matrix.to_numpy().T, atol=TOLERANCE)


def test_build_correlation_matrix_excludes_id_columns(sample_df):
    matrix = build_correlation_matrix(sample_df)
    assert "order_id" not in matrix.columns
    assert "order_id" not in matrix.index


def test_plot_correlation_heatmap_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "heatmap.png"
    plot_correlation_heatmap(sample_df, str(out))
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_pairplot
# --------------------------------------------------------------------------
def test_plot_pairplot_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "pairplot.png"
    plot_pairplot(sample_df, "product_category", str(out))
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_violin_by_category
# --------------------------------------------------------------------------
def test_plot_violin_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "violin.png"
    plot_violin_by_category(sample_df, "unit_price", "product_category", str(out))
    _assert_valid_png(out)


def test_plot_violin_handles_empty_dataframe_without_raising(tmp_path, sample_df):
    out = tmp_path / "violin_empty.png"
    empty_df = sample_df.iloc[0:0]
    plot_violin_by_category(empty_df, "unit_price", "product_category", str(out))  # must not raise
    _assert_valid_png(out)


def test_plot_violin_handles_single_category_without_raising(tmp_path, sample_df):
    out = tmp_path / "violin_single.png"
    single_cat_df = sample_df.copy()
    single_cat_df["product_category"] = "OnlyOne"
    plot_violin_by_category(single_cat_df, "unit_price", "product_category", str(out))  # must not raise
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_regression_scatter
# --------------------------------------------------------------------------
def test_plot_regression_scatter_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "regplot.png"
    plot_regression_scatter(sample_df, "quantity", "unit_price", str(out))
    _assert_valid_png(out)


def test_plot_regression_scatter_handles_empty_dataframe_without_raising(tmp_path, sample_df):
    out = tmp_path / "regplot_empty.png"
    empty_df = sample_df.iloc[0:0]
    plot_regression_scatter(empty_df, "quantity", "unit_price", str(out))  # must not raise
    _assert_valid_png(out)


# --------------------------------------------------------------------------
# plot_countplot / plot_matplotlib_vs_seaborn_comparison
# --------------------------------------------------------------------------
def test_plot_countplot_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "countplot.png"
    plot_countplot(sample_df, "product_category", str(out))
    _assert_valid_png(out)


def test_plot_matplotlib_vs_seaborn_comparison_creates_valid_png(tmp_path, sample_df):
    out = tmp_path / "comparison.png"
    plot_matplotlib_vs_seaborn_comparison(sample_df, "unit_price", "product_category", str(out))
    _assert_valid_png(out)
