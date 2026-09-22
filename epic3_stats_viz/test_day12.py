"""Pytest suite for day12_correlation_distributions.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic3_stats_viz.day12_correlation_distributions import (
        compute_zscores,
        correlation_matrix_report,
        fit_normal_distribution,
        flag_zscore_outliers,
        generate_distribution_samples,
        pearson_correlation,
        strongest_and_weakest_pairs,
    )
except ImportError:
    from day12_correlation_distributions import (
        compute_zscores,
        correlation_matrix_report,
        fit_normal_distribution,
        flag_zscore_outliers,
        generate_distribution_samples,
        pearson_correlation,
        strongest_and_weakest_pairs,
    )

TOLERANCE = 1e-6


@pytest.fixture
def perfectly_correlated():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2 * x + 1  # perfect positive linear relationship -> r should be 1.0
    return x, y


@pytest.fixture
def real_valued_pair():
    rng = np.random.default_rng(7)
    x = rng.normal(50, 10, size=100)
    y = 0.6 * x + rng.normal(0, 5, size=100)  # correlated but noisy
    return x, y


@pytest.fixture
def zscore_dataset_with_outlier():
    # A tight cluster of 30 points around 10 plus one deliberate, obvious
    # outlier at 60 - a known synthetic case, hand-checked (mean~11.6,
    # std~8.8, outlier z~5.5), where the outlier MUST be flagged at both
    # the ±2 and ±3 std thresholds and nothing else should be.
    cluster = [10.0] * 20 + [10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0, 10.1, 9.9, 10.2]
    return np.array(cluster + [60.0])


# --------------------------------------------------------------------------
# pearson_correlation
# --------------------------------------------------------------------------
def test_pearson_perfect_positive_correlation(perfectly_correlated):
    x, y = perfectly_correlated
    assert pearson_correlation(x, y) == pytest.approx(1.0, abs=TOLERANCE)


def test_pearson_perfect_negative_correlation(perfectly_correlated):
    x, y = perfectly_correlated
    assert pearson_correlation(x, -y) == pytest.approx(-1.0, abs=TOLERANCE)


def test_pearson_matches_scipy_pearsonr(real_valued_pair):
    x, y = real_valued_pair
    scratch = pearson_correlation(x, y)
    scipy_r, _ = scipy_stats.pearsonr(x, y)
    assert scratch == pytest.approx(float(scipy_r), abs=TOLERANCE)


def test_pearson_matches_numpy_corrcoef(real_valued_pair):
    x, y = real_valued_pair
    scratch = pearson_correlation(x, y)
    numpy_r = np.corrcoef(x, y)[0, 1]
    assert scratch == pytest.approx(float(numpy_r), abs=TOLERANCE)


def test_pearson_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        pearson_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_pearson_raises_on_constant_input():
    with pytest.raises(ValueError):
        pearson_correlation(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------
# correlation_matrix_report / strongest_and_weakest_pairs
# --------------------------------------------------------------------------
def test_correlation_matrix_report_excludes_id_columns():
    import pandas as pd
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    report = correlation_matrix_report(df)
    involved_columns = set(report["column_a"]) | set(report["column_b"])
    assert "order_id" not in involved_columns
    assert {"a", "b"} == involved_columns


def test_correlation_matrix_report_sorted_by_abs_pearson_desc():
    import pandas as pd
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "a": np.arange(50, dtype=float),
        "b": np.arange(50, dtype=float) * 2 + rng.normal(0, 1, 50),   # strong
        "c": rng.normal(0, 1, 50),                                     # near-zero relation to a
    })
    report = correlation_matrix_report(df)
    assert list(report["abs_pearson"]) == sorted(report["abs_pearson"], reverse=True)


def test_strongest_and_weakest_pairs_shape():
    import pandas as pd
    report = pd.DataFrame({
        "column_a": ["a", "a", "b"],
        "column_b": ["b", "c", "c"],
        "pearson": [0.9, 0.1, -0.5],
        "spearman": [0.9, 0.1, -0.5],
        "spearman_pvalue": [0.01, 0.5, 0.2],
        "abs_pearson": [0.9, 0.1, 0.5],
    }).sort_values("abs_pearson", ascending=False).reset_index(drop=True)

    extremes = strongest_and_weakest_pairs(report, n=1)
    assert extremes["strongest"].iloc[0]["column_a"] == "a"
    assert extremes["strongest"].iloc[0]["column_b"] == "b"
    assert extremes["weakest"].iloc[0]["column_a"] == "a"
    assert extremes["weakest"].iloc[0]["column_b"] == "c"


# --------------------------------------------------------------------------
# distribution samples
# --------------------------------------------------------------------------
def test_generate_distribution_samples_shapes_and_keys():
    result = generate_distribution_samples(size=500, seed=1)
    assert set(result.keys()) == {"normal", "binomial", "poisson"}
    for info in result.values():
        assert len(info["samples"]) == 500


def test_generate_distribution_samples_empirical_close_to_theoretical():
    # With 5000 samples, empirical mean should land close to the
    # theoretical mean for all three distributions (loose tolerance -
    # this is a statistical sanity check, not an exact-match test).
    result = generate_distribution_samples(size=5000, seed=1)
    for name, info in result.items():
        assert info["empirical_mean"] == pytest.approx(info["theoretical_mean"], rel=0.1), name


# --------------------------------------------------------------------------
# fit_normal_distribution
# --------------------------------------------------------------------------
def test_fit_normal_distribution_detects_normal_data():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100, scale=15, size=1000)
    result = fit_normal_distribution(data)
    assert result["mean"] == pytest.approx(100, abs=1.5)
    assert result["std"] == pytest.approx(15, abs=1.5)
    assert result["is_normal"] is True


def test_fit_normal_distribution_rejects_non_normal_data():
    rng = np.random.default_rng(42)
    # Strongly right-skewed exponential data should fail the Shapiro-Wilk
    # normality test at alpha=0.05.
    data = rng.exponential(scale=2.0, size=500)
    result = fit_normal_distribution(data)
    assert result["is_normal"] is False
    assert result["shapiro_pvalue"] <= result["alpha"]


# --------------------------------------------------------------------------
# compute_zscores / flag_zscore_outliers - against a known synthetic dataset
# --------------------------------------------------------------------------
def test_compute_zscores_mean_and_std_of_output():
    rng = np.random.default_rng(5)
    data = rng.normal(20, 4, size=200)
    z = compute_zscores(data)
    # Z-scores always have mean ~0 and std ~1, by construction.
    assert np.mean(z) == pytest.approx(0.0, abs=1e-9)
    assert np.std(z, ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_compute_zscores_raises_on_constant_data():
    with pytest.raises(ValueError):
        compute_zscores(np.array([5.0, 5.0, 5.0, 5.0]))


def test_flag_zscore_outliers_known_synthetic_case(zscore_dataset_with_outlier):
    result = flag_zscore_outliers(zscore_dataset_with_outlier, thresholds=(2.0, 3.0))
    assert result["beyond_2.0_std"]["count"] == 1
    assert result["beyond_3.0_std"]["count"] == 1
    assert result["beyond_2.0_std"]["values"] == [60.0]


def test_flag_zscore_outliers_no_outliers_in_tight_cluster():
    tight_data = np.array([10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9])
    result = flag_zscore_outliers(tight_data, thresholds=(2.0, 3.0))
    assert result["beyond_2.0_std"]["count"] == 0
    assert result["beyond_3.0_std"]["count"] == 0
