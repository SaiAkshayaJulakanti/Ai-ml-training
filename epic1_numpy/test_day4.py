"""Pytest suite for day4_vectorization_math.py.

Run with: pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic1_numpy.day4_vectorization_math import (
        benchmark_loop_vs_vectorized,
        dot_product_loop,
        dot_product_vectorized,
        elementwise_distance_loop,
        elementwise_distance_vectorized,
        euclidean_distance_matrix,
        matrix_ops_report,
        statistics_demo,
        sum_of_squares_loop,
        sum_of_squares_vectorized,
    )
except ImportError:
    from day4_vectorization_math import (
        benchmark_loop_vs_vectorized,
        dot_product_loop,
        dot_product_vectorized,
        elementwise_distance_loop,
        elementwise_distance_vectorized,
        euclidean_distance_matrix,
        matrix_ops_report,
        statistics_demo,
        sum_of_squares_loop,
        sum_of_squares_vectorized,
    )


# --------------------------------------------------------------------------
# Loop vs vectorized equivalence (np.allclose, per the spec)
# --------------------------------------------------------------------------
def test_sum_of_squares_loop_matches_vectorized():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    loop_result = sum_of_squares_loop(arr)
    vec_result = sum_of_squares_vectorized(arr)
    assert np.allclose(loop_result, vec_result)


def test_dot_product_loop_matches_vectorized():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert np.allclose(dot_product_loop(a, b), dot_product_vectorized(a, b))


def test_elementwise_distance_loop_matches_vectorized():
    a = np.array([1.0, 5.0, 3.0])
    b = np.array([4.0, 2.0, 3.0])
    loop_result = elementwise_distance_loop(a, b)
    vec_result = elementwise_distance_vectorized(a, b)
    assert np.allclose(loop_result, vec_result)


# --------------------------------------------------------------------------
# benchmark_loop_vs_vectorized
# --------------------------------------------------------------------------
def test_benchmark_returns_expected_keys():
    results = benchmark_loop_vs_vectorized(1000)
    assert set(results.keys()) == {"sum_of_squares", "dot_product", "elementwise_distance"}
    for op_results in results.values():
        assert set(op_results.keys()) == {"loop_time", "vectorized_time", "speedup"}
        assert op_results["loop_time"] >= 0
        assert op_results["vectorized_time"] >= 0


def test_benchmark_vectorized_is_faster():
    """The vectorized version should be faster than the loop version - the whole point of Day 4."""
    results = benchmark_loop_vs_vectorized(50_000)
    for op_name, timings in results.items():
        assert timings["speedup"] > 1.0, f"{op_name} was not faster when vectorized"


# --------------------------------------------------------------------------
# matrix_ops_report - normal and edge cases
# --------------------------------------------------------------------------
def test_matrix_ops_report_square_invertible():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    report = matrix_ops_report(a, b)
    assert report["errors"] == []
    assert np.allclose(report["product"], a @ b)
    assert np.allclose(report["transpose_a"], a.T)
    assert report["det_a"] is not None
    assert report["inverse_a"] is not None
    assert np.allclose(a @ report["inverse_a"], np.eye(2), atol=1e-8)


def test_matrix_ops_report_non_invertible_handled_gracefully():
    """Edge case: a singular (non-invertible) matrix must be caught, not crash the program."""
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])   # det = 0
    other = np.array([[1.0, 0.0], [0.0, 1.0]])
    report = matrix_ops_report(singular, other)
    assert report["inverse_a"] is None
    assert any("not invertible" in err for err in report["errors"])


def test_matrix_ops_report_non_square_handled_gracefully():
    """Edge case: a non-square matrix has no determinant/inverse - must be reported, not crash."""
    non_square = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    square = np.array([[1.0, 0.0], [0.0, 1.0]])
    report = matrix_ops_report(non_square, square)
    assert report["det_a"] is None
    assert report["inverse_a"] is None
    assert any("not square" in err for err in report["errors"])


def test_matrix_ops_report_incompatible_product_shapes():
    """Edge case: shapes that can't be matrix-multiplied should be reported, not crash."""
    a = np.array([[1.0, 2.0]])       # shape (1, 2)
    b = np.array([[1.0, 2.0]])       # shape (1, 2) - incompatible for a @ b
    report = matrix_ops_report(a, b)
    assert report["product"] is None
    assert any("product" in err for err in report["errors"])


# --------------------------------------------------------------------------
# euclidean_distance_matrix
# --------------------------------------------------------------------------
def test_euclidean_distance_matrix_known_values():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[0.0, 0.0], [3.0, 4.0]])
    result = euclidean_distance_matrix(a, b)
    expected = np.array([
        [0.0, 5.0],
        [np.sqrt(2), np.sqrt(13)],
    ])
    assert np.allclose(result, expected)


def test_euclidean_distance_matrix_shape():
    a = np.random.rand(3, 4)
    b = np.random.rand(5, 4)
    result = euclidean_distance_matrix(a, b)
    assert result.shape == (3, 5)


def test_euclidean_distance_matrix_mismatched_dims_raises():
    a = np.random.rand(3, 4)
    b = np.random.rand(5, 2)   # different number of columns
    with pytest.raises(ValueError):
        euclidean_distance_matrix(a, b)


# --------------------------------------------------------------------------
# statistics_demo
# --------------------------------------------------------------------------
def test_statistics_demo_matches_numpy_directly():
    arr = np.array([[10.0, 200.0], [20.0, 250.0], [30.0, 300.0]])
    stats = statistics_demo(arr)
    assert np.isclose(stats["mean_overall"], np.mean(arr))
    assert np.allclose(stats["mean_per_column"], np.mean(arr, axis=0))
    assert np.isclose(stats["std_overall"], np.std(arr))
    assert np.allclose(stats["var_per_column"], np.var(arr, axis=0))
