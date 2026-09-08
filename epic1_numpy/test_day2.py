"""Pytest suite for day2_numpy_basics.py.

Run with: pytest -v
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

file_dir = Path(__file__).parent
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic1_numpy.day2_numpy_basics import (
        create_identity_matrix,
        dtype_cast_demo,
        flatten_vs_ravel_demo,
        hstack_demo,
        inspect_array,
        random_matrix_stats,
        reshape_pipeline,
        stack_arrays,
        vstack_demo,
    )
except ImportError:
    from day2_numpy_basics import (
        create_identity_matrix,
        dtype_cast_demo,
        flatten_vs_ravel_demo,
        hstack_demo,
        inspect_array,
        random_matrix_stats,
        reshape_pipeline,
        stack_arrays,
        vstack_demo,
    )


# --------------------------------------------------------------------------
# Identity matrix
# --------------------------------------------------------------------------
def test_create_identity_matrix_n1():
    """Edge case: n=1 should give a 1x1 matrix containing just [[1.]]."""
    result = create_identity_matrix(1)
    assert result.shape == (1, 1)
    assert np.array_equal(result, np.array([[1.0]]))


def test_create_identity_matrix_n5():
    result = create_identity_matrix(5)
    assert result.shape == (5, 5)
    assert np.array_equal(result, np.eye(5))
    # Diagonal is all 1s, everything else is 0
    assert np.trace(result) == 5
    assert result.sum() == 5


# --------------------------------------------------------------------------
# random_matrix_stats
# --------------------------------------------------------------------------
def test_random_matrix_stats_keys_and_ranges():
    stats = random_matrix_stats(10, 10)
    assert set(stats.keys()) == {"min", "max", "mean", "std"}
    assert 0.0 <= stats["min"] <= 1.0
    assert 0.0 <= stats["max"] <= 1.0
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["std"] >= 0.0


def test_random_matrix_stats_single_element():
    """Edge case: a 1x1 matrix has zero standard deviation and min == max == mean."""
    stats = random_matrix_stats(1, 1)
    assert stats["min"] == stats["max"] == stats["mean"]
    assert stats["std"] == 0.0


# --------------------------------------------------------------------------
# reshape_pipeline
# --------------------------------------------------------------------------
def test_reshape_pipeline_valid_shape():
    arr = np.arange(12)
    reshaped = reshape_pipeline(arr, (3, 4))
    assert reshaped.shape == (3, 4)
    assert np.array_equal(reshaped, arr.reshape(3, 4))


def test_reshape_pipeline_invalid_shape_raises():
    """Invalid reshape (incompatible element count) must raise ValueError."""
    arr = np.arange(12)
    with pytest.raises(ValueError):
        reshape_pipeline(arr, (5, 5))


def test_reshape_pipeline_preserves_data_order():
    arr = np.arange(6)
    reshaped = reshape_pipeline(arr, (2, 3))
    assert reshaped.tolist() == [[0, 1, 2], [3, 4, 5]]


# --------------------------------------------------------------------------
# flatten vs ravel
# --------------------------------------------------------------------------
def test_flatten_is_a_copy():
    arr = np.arange(6).reshape(2, 3)
    result = flatten_vs_ravel_demo(arr)
    assert result["flatten_is_copy"] is True


def test_ravel_is_a_view():
    arr = np.arange(6).reshape(2, 3)
    result = flatten_vs_ravel_demo(arr)
    assert result["ravel_is_view"] is True


# --------------------------------------------------------------------------
# stack_arrays / vstack / hstack
# --------------------------------------------------------------------------
def test_stack_arrays_axis0():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = stack_arrays([a, b], axis=0)
    assert result.shape == (4, 2)
    assert np.array_equal(result, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))


def test_stack_arrays_incompatible_shapes_raises():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6, 7]])
    with pytest.raises(ValueError):
        stack_arrays([a, b], axis=0)


def test_vstack_demo():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = vstack_demo([a, b])
    assert result.shape == (2, 3)


def test_hstack_demo():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = hstack_demo([a, b])
    assert result.shape == (6,)
    assert result.tolist() == [1, 2, 3, 4, 5, 6]


# --------------------------------------------------------------------------
# inspect_array
# --------------------------------------------------------------------------
def test_inspect_array_1d():
    arr = np.arange(10)
    info = inspect_array(arr)
    assert info["shape"] == (10,)
    assert info["ndim"] == 1
    assert info["size"] == 10


def test_inspect_array_2d():
    arr = np.ones((3, 4))
    info = inspect_array(arr)
    assert info["shape"] == (3, 4)
    assert info["ndim"] == 2
    assert info["size"] == 12


def test_inspect_array_3d():
    arr = np.zeros((2, 3, 4))
    info = inspect_array(arr)
    assert info["shape"] == (2, 3, 4)
    assert info["ndim"] == 3
    assert info["size"] == 24
    assert info["total_bytes"] == info["size"] * info["itemsize"]


# --------------------------------------------------------------------------
# dtype casting
# --------------------------------------------------------------------------
def test_dtype_cast_int32_to_float64_doubles_memory():
    arr = np.ones((10, 10), dtype=np.int32)
    casted = dtype_cast_demo(arr, "float64")
    assert casted.dtype == np.float64
    assert casted.nbytes == arr.nbytes * 2


def test_dtype_cast_preserves_values():
    arr = np.array([1, 2, 3], dtype=np.int32)
    casted = dtype_cast_demo(arr, "float64")
    assert casted.tolist() == [1.0, 2.0, 3.0]
