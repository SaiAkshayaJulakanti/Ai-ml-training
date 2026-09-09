"""Pytest suite for day3_indexing_broadcasting.py.

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
    from epic1_numpy.day3_indexing_broadcasting import (
        extract_submatrix,
        filter_outliers,
        normalize_broadcast,
        select_rows_by_index,
    )
except ImportError:
    from day3_indexing_broadcasting import (
        extract_submatrix,
        filter_outliers,
        normalize_broadcast,
        select_rows_by_index,
    )


# --------------------------------------------------------------------------
# filter_outliers
# --------------------------------------------------------------------------
def test_filter_outliers_removes_far_values():
    arr = np.array([10, 11, 9, 10, 100, 12, -80, 11])
    result = filter_outliers(arr, threshold=15)
    assert 100 not in result
    assert -80 not in result
    assert 10 in result


def test_filter_outliers_no_outliers():
    """Edge case: every value is close to the mean - nothing should be removed."""
    arr = np.array([10, 11, 9, 10, 12, 11])
    result = filter_outliers(arr, threshold=100)
    assert len(result) == len(arr)
    assert np.array_equal(result, arr)


def test_filter_outliers_all_outliers():
    """Edge case: threshold is so small that everything counts as an outlier."""
    arr = np.array([10, 11, 9, 10, 12, 11])
    result = filter_outliers(arr, threshold=0.0001)
    # mean is not exactly any single value here, so almost everything is filtered;
    # at minimum, result should be strictly smaller than the original
    assert len(result) < len(arr)


def test_filter_outliers_empty_array():
    """Edge case: empty array should return an empty array without error."""
    arr = np.array([])
    result = filter_outliers(arr, threshold=5)
    assert len(result) == 0


# --------------------------------------------------------------------------
# extract_submatrix
# --------------------------------------------------------------------------
def test_extract_submatrix_basic():
    arr = np.arange(20).reshape(4, 5)
    result = extract_submatrix(arr, row_range=(1, 3), col_range=(0, 2))
    expected = np.array([[5, 6], [10, 11]])
    assert np.array_equal(result, expected)


def test_extract_submatrix_full_array():
    arr = np.arange(12).reshape(3, 4)
    result = extract_submatrix(arr, row_range=(0, 3), col_range=(0, 4))
    assert np.array_equal(result, arr)


def test_extract_submatrix_is_a_view():
    """Basic slicing returns a view - mutating the result should affect the original."""
    arr = np.arange(12).reshape(3, 4)
    sub = extract_submatrix(arr, row_range=(0, 1), col_range=(0, 1))
    sub[0, 0] = -999
    assert arr[0, 0] == -999


def test_extract_submatrix_out_of_bounds_raises():
    """Edge case: a range starting outside the array's bounds must raise, not silently return empty."""
    arr = np.arange(12).reshape(3, 4)
    with pytest.raises(ValueError):
        extract_submatrix(arr, row_range=(10, 12), col_range=(0, 2))


def test_extract_submatrix_rejects_non_2d():
    arr = np.arange(10)
    with pytest.raises(ValueError):
        extract_submatrix(arr, row_range=(0, 1), col_range=(0, 1))


# --------------------------------------------------------------------------
# select_rows_by_index (fancy indexing)
# --------------------------------------------------------------------------
def test_select_rows_by_index_basic():
    arr = np.arange(20).reshape(4, 5)
    result = select_rows_by_index(arr, [3, 0])
    expected = np.array([[15, 16, 17, 18, 19], [0, 1, 2, 3, 4]])
    assert np.array_equal(result, expected)


def test_select_rows_by_index_is_a_copy():
    """Fancy indexing returns a copy - mutating the result must NOT affect the original."""
    arr = np.arange(20).reshape(4, 5)
    selected = select_rows_by_index(arr, [0])
    selected[0, 0] = -999
    assert arr[0, 0] == 0


def test_select_rows_by_index_duplicates_allowed():
    arr = np.arange(12).reshape(3, 4)
    result = select_rows_by_index(arr, [1, 1, 0])
    assert result.shape == (3, 4)
    assert np.array_equal(result[0], result[1])


def test_select_rows_by_index_out_of_bounds_raises():
    """Edge case: an index beyond the array's row count must raise a clear ValueError."""
    arr = np.arange(12).reshape(3, 4)
    with pytest.raises(ValueError):
        select_rows_by_index(arr, [0, 99])


def test_select_rows_by_index_rejects_non_2d():
    arr = np.arange(10)
    with pytest.raises(ValueError):
        select_rows_by_index(arr, [0])


# --------------------------------------------------------------------------
# normalize_broadcast
# --------------------------------------------------------------------------
def test_normalize_broadcast_mean_and_std():
    arr = np.array([
        [10.0, 200.0, 1.0],
        [20.0, 250.0, 2.0],
        [30.0, 300.0, 3.0],
        [40.0, 350.0, 4.0],
    ])
    result = normalize_broadcast(arr)
    assert np.allclose(result.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(result.std(axis=0), 1.0, atol=1e-8)


def test_normalize_broadcast_preserves_shape():
    arr = np.random.rand(5, 3)
    result = normalize_broadcast(arr)
    assert result.shape == arr.shape


def test_normalize_broadcast_rejects_non_2d():
    """Edge case: a 1D array should raise ValueError, not silently misbehave."""
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        normalize_broadcast(arr)


def test_normalize_broadcast_zero_std_column_raises():
    """Edge case: a column of identical values has std=0 - must raise, not divide by zero."""
    arr = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    with pytest.raises(ValueError):
        normalize_broadcast(arr)


# --------------------------------------------------------------------------
# Broadcasting failure case
# --------------------------------------------------------------------------
def test_broadcasting_mismatched_shapes_raises():
    """Mismatched shapes that cannot broadcast must raise ValueError."""
    a = np.ones((3, 4))
    b = np.ones((3, 2))
    with pytest.raises(ValueError):
        a + b


def test_broadcasting_compatible_shapes_succeeds():
    """Sanity check: a genuinely compatible shape pair should NOT raise."""
    a = np.ones((3, 4))
    b = np.ones((4,))
    result = a + b
    assert result.shape == (3, 4)
