"""Pytest suite for day11_descriptive_stats.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic3_stats_viz.day11_descriptive_stats import (
        bayes_theorem,
        compute_central_tendency,
        compute_spread,
        conditional_probability,
        simple_probability,
        spam_bayes_worked_example,
    )
except ImportError:
    from day11_descriptive_stats import (
        bayes_theorem,
        compute_central_tendency,
        compute_spread,
        conditional_probability,
        simple_probability,
        spam_bayes_worked_example,
    )

TOLERANCE = 1e-6


@pytest.fixture
def numeric_data() -> np.ndarray:
    # Deliberately uneven length, a repeated value (for mode), and a mild
    # outlier so range/IQR are non-trivial.
    return np.array([12.0, 15.0, 12.0, 20.0, 18.0, 25.0, 12.0, 30.0, 22.0, 100.0])


@pytest.fixture
def prob_df() -> pd.DataFrame:
    return pd.DataFrame({
        "is_high_value": [True, True, False, False, True, False, True, False],
        "is_electronics": [True, False, False, True, True, False, False, False],
    })


# --------------------------------------------------------------------------
# compute_central_tendency
# --------------------------------------------------------------------------
def test_mean_matches_numpy(numeric_data):
    result = compute_central_tendency(numeric_data)
    assert result["mean"] == pytest.approx(float(np.mean(numeric_data)), abs=TOLERANCE)


def test_median_matches_numpy(numeric_data):
    result = compute_central_tendency(numeric_data)
    assert result["median"] == pytest.approx(float(np.median(numeric_data)), abs=TOLERANCE)


def test_mode_matches_scipy(numeric_data):
    result = compute_central_tendency(numeric_data)
    scipy_mode = scipy_stats.mode(numeric_data, keepdims=True).mode[0]
    assert result["mode"] == pytest.approx(float(scipy_mode), abs=TOLERANCE)
    assert result["mode"] == 12.0  # 12.0 appears 3 times, the most of any value


def test_central_tendency_raises_on_empty():
    with pytest.raises(ValueError):
        compute_central_tendency(np.array([]))


# --------------------------------------------------------------------------
# compute_spread
# --------------------------------------------------------------------------
def test_population_variance_matches_numpy_ddof0(numeric_data):
    result = compute_spread(numeric_data)
    assert result["variance_population"] == pytest.approx(float(np.var(numeric_data, ddof=0)), abs=TOLERANCE)
    assert result["std_population"] == pytest.approx(float(np.std(numeric_data, ddof=0)), abs=TOLERANCE)


def test_sample_variance_matches_pandas_ddof1(numeric_data):
    result = compute_spread(numeric_data)
    series = pd.Series(numeric_data)
    assert result["variance_sample"] == pytest.approx(float(series.var(ddof=1)), abs=TOLERANCE)
    assert result["std_sample"] == pytest.approx(float(series.std(ddof=1)), abs=TOLERANCE)


def test_sample_variance_greater_than_population_variance(numeric_data):
    # Dividing by (n-1) instead of n always yields a variance >= the
    # population version for n > 1 - this is the whole point of Bessel's
    # correction.
    result = compute_spread(numeric_data)
    assert result["variance_sample"] > result["variance_population"]


def test_range_matches_numpy_ptp(numeric_data):
    result = compute_spread(numeric_data)
    assert result["range"] == pytest.approx(float(np.ptp(numeric_data)), abs=TOLERANCE)


def test_iqr_matches_scipy(numeric_data):
    result = compute_spread(numeric_data)
    assert result["iqr"] == pytest.approx(float(scipy_stats.iqr(numeric_data)), abs=TOLERANCE)


def test_spread_raises_on_too_few_points():
    with pytest.raises(ValueError):
        compute_spread(np.array([5.0]))


# --------------------------------------------------------------------------
# probability functions
# --------------------------------------------------------------------------
def test_simple_probability(prob_df):
    # 4 of 8 rows have is_high_value == True
    assert simple_probability(prob_df, "is_high_value") == pytest.approx(0.5, abs=TOLERANCE)


def test_conditional_probability_manual_check(prob_df):
    # Manually: is_electronics True for rows 0,3,4 (3 rows).
    # Of those, is_high_value True for rows 0 and 4 -> 2/3.
    expected = 2 / 3
    result = conditional_probability(prob_df, "is_high_value", "is_electronics")
    assert result == pytest.approx(expected, abs=TOLERANCE)


def test_conditional_probability_raises_on_missing_column(prob_df):
    with pytest.raises(ValueError):
        conditional_probability(prob_df, "does_not_exist", "is_electronics")


def test_conditional_probability_raises_when_p_b_zero(prob_df):
    prob_df = prob_df.copy()
    prob_df["never_true"] = False
    with pytest.raises(ValueError):
        conditional_probability(prob_df, "is_high_value", "never_true")


# --------------------------------------------------------------------------
# bayes_theorem - validated against a manually verified textbook example
# --------------------------------------------------------------------------
def test_bayes_theorem_textbook_disease_example():
    # Classic textbook example: a disease with 1% prevalence, a test with
    # 99% sensitivity (true positive rate) and a 5% false positive rate.
    # P(Disease) = 0.01, P(+|Disease) = 0.99, P(+|No Disease) = 0.05
    # P(+) = 0.99*0.01 + 0.05*0.99 = 0.0594
    # P(Disease|+) = 0.99*0.01 / 0.0594 ~= 0.1667 (manually verified)
    p_disease = 0.01
    p_pos_given_disease = 0.99
    p_pos_given_no_disease = 0.05
    p_no_disease = 1 - p_disease

    p_pos = p_pos_given_disease * p_disease + p_pos_given_no_disease * p_no_disease
    result = bayes_theorem(p_a=p_disease, p_b_given_a=p_pos_given_disease, p_b=p_pos)

    assert result == pytest.approx(0.16666667, abs=1e-6)


def test_bayes_theorem_certain_evidence_returns_prior():
    # If P(B|A) == P(B) == 1, the posterior should equal the prior exactly.
    result = bayes_theorem(p_a=0.3, p_b_given_a=1.0, p_b=1.0)
    assert result == pytest.approx(0.3, abs=TOLERANCE)


def test_bayes_theorem_raises_on_invalid_probability():
    with pytest.raises(ValueError):
        bayes_theorem(p_a=1.5, p_b_given_a=0.5, p_b=0.5)


def test_bayes_theorem_raises_on_zero_evidence():
    with pytest.raises(ValueError):
        bayes_theorem(p_a=0.5, p_b_given_a=0.5, p_b=0.0)


# --------------------------------------------------------------------------
# worked spam example - sanity checks on the full pipeline
# --------------------------------------------------------------------------
def test_spam_worked_example_posterior_matches_manual_calc():
    result = spam_bayes_worked_example()
    # p_spam=0.2, p_free_given_spam=0.8, p_free_given_ham=0.05, p_ham=0.8
    # p_free = 0.8*0.2 + 0.05*0.8 = 0.2
    # p_spam_given_free = 0.8*0.2/0.2 = 0.8
    assert result["p_free"] == pytest.approx(0.2, abs=TOLERANCE)
    assert result["p_spam_given_free"] == pytest.approx(0.8, abs=TOLERANCE)


def test_spam_worked_example_posterior_exceeds_prior():
    # Seeing the word "free" should raise the probability of spam above
    # the baseline (unconditional) rate of spam.
    result = spam_bayes_worked_example()
    assert result["p_spam_given_free"] > result["p_spam"]
