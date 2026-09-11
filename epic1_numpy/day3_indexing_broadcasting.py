"""Day 3: Indexing, Slicing & Broadcasting

This module covers basic slicing, boolean masking, fancy indexing, and
NumPy's broadcasting rules - the foundation for vectorized data
manipulation used throughout ML pipelines.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Needed to import the shared common/numpy_utils.py module (used by
# normalize_broadcast below, refactored here as of the Day 5 project).
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from common.numpy_utils import zscore_normalize
except ImportError:
    # Fallback so this file still runs standalone even if common/ isn't
    # on the path for some reason (e.g. Day 3 run in isolation before
    # Day 5's common/ module existed in an older checkout).
    def zscore_normalize(arr: np.ndarray, axis: int = 0) -> np.ndarray:
        mean = arr.mean(axis=axis)
        std = arr.std(axis=axis)
        if np.any(std == 0):
            raise ValueError(
                "Cannot normalize: at least one slice has zero standard "
                "deviation (all identical values), which would divide by zero."
            )
        return (arr - mean) / std

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Basic slicing on 1D/2D arrays
# --------------------------------------------------------------------------
def slicing_demo() -> None:
    """Demonstrate basic slicing on 1D and 2D arrays: rows, columns, sub-matrices.

    Slicing syntax is arr[start:stop:step] per axis, separated by commas
    for multi-dimensional arrays. Unlike fancy indexing (Section 3), a
    basic slice always returns a VIEW into the original array - not a copy.
    """
    arr_1d = np.arange(10)
    logger.info("1D array: %s", arr_1d)
    logger.info("arr_1d[2:7]      -> %s  (elements at index 2 through 6)", arr_1d[2:7])
    logger.info("arr_1d[::2]      -> %s  (every 2nd element)", arr_1d[::2])
    logger.info("arr_1d[::-1]     -> %s  (reversed)", arr_1d[::-1])

    arr_2d = np.arange(12).reshape(3, 4)
    logger.info("2D array:\n%s", arr_2d)
    logger.info("arr_2d[1, :]     -> %s  (row 1, all columns)", arr_2d[1, :])
    logger.info("arr_2d[:, 2]     -> %s  (all rows, column 2)", arr_2d[:, 2])
    logger.info("arr_2d[0:2, 1:3] -> sub-matrix (rows 0-1, cols 1-2):\n%s", arr_2d[0:2, 1:3])

    # Proof that basic slicing returns a VIEW, not a copy:
    row_view = arr_2d[1, :]
    row_view[0] = -1
    logger.info(
        "After mutating the row-slice view, arr_2d[1, 0] is now: %d (view, not copy)",
        arr_2d[1, 0],
    )


# --------------------------------------------------------------------------
# 2. Boolean masking
# --------------------------------------------------------------------------
def boolean_mask_demo(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Return only the elements of `arr` greater than `threshold`, using a boolean mask.

    A boolean mask is an array of True/False values (same shape as `arr`)
    produced by a comparison like `arr > threshold`. Indexing `arr` with
    that mask (`arr[mask]`) pulls out only the True positions, ALWAYS
    returning a flat 1D array (even if `arr` was multi-dimensional) and
    ALWAYS returning a copy, not a view.
    """
    mask = arr > threshold
    return arr[mask]


def filter_outliers(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Filter out values whose absolute deviation from the mean exceeds `threshold`.

    Uses boolean masking: an element is kept only if
    |element - mean| <= threshold.

    Args:
        arr: Input array (any shape - flattened internally for the check).
        threshold: Maximum allowed absolute deviation from the mean.

    Returns:
        A 1D array containing only the non-outlier elements, in their
        original relative order.
    """
    flat = arr.flatten()
    if flat.size == 0:
        return flat
    mean = flat.mean()
    mask = np.abs(flat - mean) <= threshold
    return flat[mask]


# --------------------------------------------------------------------------
# 3. Fancy indexing
# --------------------------------------------------------------------------
def fancy_indexing_demo(arr: np.ndarray, row_indices: List[int], col_indices: List[int]) -> None:
    """Demonstrate fancy indexing: selecting specific rows/columns via index arrays.

    Fancy indexing (indexing with a list/array of integers, rather than a
    slice) ALWAYS returns a COPY, never a view - unlike basic slicing.
    This matters if you plan to mutate the result and don't want it to
    affect the original array.
    """
    logger.info("Original array:\n%s", arr)
    logger.info("arr[%s]        -> selected rows:\n%s", row_indices, arr[row_indices])
    logger.info("arr[:, %s]     -> selected columns:\n%s", col_indices, arr[:, col_indices])
    logger.info(
        "arr[%s][:, %s] -> selected rows AND columns:\n%s",
        row_indices, col_indices, arr[row_indices][:, col_indices],
    )


def select_rows_by_index(arr: np.ndarray, indices: List[int]) -> np.ndarray:
    """Select specific rows of a 2D array using fancy indexing.

    Args:
        arr: A 2D array.
        indices: List of row indices to select, in the desired order
            (duplicates and out-of-original-order indices are both allowed).

    Returns:
        A new array (copy) containing only the selected rows, in the
        order given by `indices`.

    Raises:
        ValueError: If `arr` is not 2D, or if any index in `indices` is
            out of bounds for `arr`'s row count.
    """
    if arr.ndim != 2:
        raise ValueError(f"select_rows_by_index requires a 2D array, got ndim={arr.ndim}")

    n_rows = arr.shape[0]
    out_of_bounds = [i for i in indices if i < -n_rows or i >= n_rows]
    if out_of_bounds:
        raise ValueError(
            f"Row index/indices {out_of_bounds} out of bounds for array "
            f"with {n_rows} rows (valid range: {-n_rows} to {n_rows - 1})."
        )

    return arr[indices]


def extract_submatrix(
    arr: np.ndarray, row_range: Tuple[int, int], col_range: Tuple[int, int]
) -> np.ndarray:
    """Extract a sub-matrix from `arr` using basic slicing (a view, not a copy).

    Args:
        arr: A 2D array.
        row_range: (start, stop) row bounds, stop exclusive.
        col_range: (start, stop) column bounds, stop exclusive.

    Returns:
        The sliced sub-matrix (a view into `arr`).

    Raises:
        ValueError: If `arr` is not 2D, or if the requested range is
            fully outside the array's actual bounds (NumPy would
            otherwise silently return an empty slice instead of
            signaling the mistake).
    """
    if arr.ndim != 2:
        raise ValueError(f"extract_submatrix requires a 2D array, got ndim={arr.ndim}")

    n_rows, n_cols = arr.shape
    row_start, row_stop = row_range
    col_start, col_stop = col_range

    if row_start >= n_rows or col_start >= n_cols:
        raise ValueError(
            f"Requested range rows={row_range}, cols={col_range} starts "
            f"outside array bounds (array shape is {arr.shape})."
        )

    return arr[row_start:row_stop, col_start:col_stop]


# --------------------------------------------------------------------------
# 4. Broadcasting
# --------------------------------------------------------------------------
def broadcasting_demo() -> None:
    """Demonstrate broadcasting: scalar+array, 1D+2D, and a mismatched-shape failure.

    BROADCASTING RULES (how NumPy decides if two shapes can combine):
    Compare shapes element-wise starting from the TRAILING (rightmost)
    dimension. Two dimensions are compatible if:
        (a) they are equal, OR
        (b) one of them is 1 (it gets "stretched" to match the other)
    If one array has fewer dimensions, it's padded with 1s on the LEFT
    until both have the same number of dimensions, then the same rule
    applies. If any dimension pair fails both (a) and (b), broadcasting
    is impossible and NumPy raises a ValueError.

    Worked example 1 - scalar + array (shapes: () and (3,)):
        A scalar has 0 dimensions - treat it as shape (1,), which
        broadcasts against ANY shape by rule (b).
        [1, 2, 3] + 10 -> [11, 12, 13]

    Worked example 2 - 1D + 2D (shapes: (4,) and (3, 4)):
        (4,) is padded on the left to (1, 4).
        Compare to (3, 4): trailing dim 4==4 (rule a), leading dim 1
        stretches to 3 (rule b). Result shape: (3, 4).
        Each row of the 2D array gets the 1D array added to it.

    Worked example 3 - mismatched shapes that CANNOT broadcast:
        Shapes (3, 4) and (3, 2): trailing dims are 4 vs 2 - neither
        equal, nor is either equal to 1. Broadcasting rule (a)/(b) both
        fail -> ValueError: operands could not be broadcast together.
    """
    # --- Worked example 1: scalar + array ---
    arr = np.array([1, 2, 3])
    result = arr + 10
    logger.info("scalar + array: %s + 10 = %s", arr, result)

    # --- Worked example 2: 1D + 2D ---
    row_vector = np.array([10, 20, 30, 40])       # shape (4,)
    matrix = np.arange(12).reshape(3, 4)           # shape (3, 4)
    result_2d = matrix + row_vector
    logger.info("1D (%s) + 2D (%s):\n%s", row_vector.shape, matrix.shape, result_2d)

    # --- Worked example 3: mismatched shapes that CANNOT broadcast ---
    bad_a = np.ones((3, 4))
    bad_b = np.ones((3, 2))
    try:
        bad_a + bad_b
    except ValueError as e:
        logger.info(
            "Expected broadcasting failure for shapes %s and %s: %s",
            bad_a.shape, bad_b.shape, e,
        )


def normalize_broadcast(arr: np.ndarray) -> np.ndarray:
    """Normalize each column of a 2D array to mean 0, std 1 - using broadcasting only.

    For each column: (column - column_mean) / column_std.

    This works with ZERO explicit Python loops. `arr.mean(axis=0)` and
    `arr.std(axis=0)` each return a 1D array of shape (n_cols,) - one
    value per column. Subtracting/dividing that 1D array from the full
    2D array relies entirely on broadcasting: shape (n_cols,) is padded
    to (1, n_cols), which then stretches to match (n_rows, n_cols) - the
    same "worked example 2" pattern from broadcasting_demo() above.

    Args:
        arr: A 2D array of shape (n_rows, n_cols).

    Returns:
        A new array of the same shape, normalized column-wise.

    Raises:
        ValueError: If `arr` is not 2D, or if any column has zero
            standard deviation (which would cause a division by zero).
    """
    if arr.ndim != 2:
        raise ValueError(f"normalize_broadcast requires a 2D array, got ndim={arr.ndim}")

    # The actual normalization math (mean/std + broadcasting, with the
    # zero-std ValueError guard) is refactored into common/numpy_utils.py
    # as of the Day 5 project, so it has one single implementation shared
    # across the whole training repo instead of being duplicated here.
    return zscore_normalize(arr, axis=0)


# --------------------------------------------------------------------------
# 5. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 3: Indexing, Slicing & Broadcasting ===")

    logger.info("--- Basic slicing demo ---")
    slicing_demo()

    logger.info("--- Boolean masking demo ---")
    sample = np.array([5, 12, 3, 27, 8, 1, 19])
    logger.info("Array: %s", sample)
    logger.info("Elements > 10: %s", boolean_mask_demo(sample, 10))

    logger.info("--- filter_outliers demo ---")
    data_with_outliers = np.array([10, 11, 9, 10, 100, 12, -80, 11])
    logger.info("Original: %s", data_with_outliers)
    logger.info("Filtered (threshold=15): %s", filter_outliers(data_with_outliers, 15))

    logger.info("--- Fancy indexing demo ---")
    matrix = np.arange(20).reshape(4, 5)
    fancy_indexing_demo(matrix, row_indices=[0, 2], col_indices=[1, 3])

    logger.info("--- select_rows_by_index demo ---")
    logger.info("select_rows_by_index(matrix, [3, 0]):\n%s", select_rows_by_index(matrix, [3, 0]))

    logger.info("--- extract_submatrix demo ---")
    logger.info(
        "extract_submatrix(matrix, (1, 3), (0, 2)):\n%s",
        extract_submatrix(matrix, (1, 3), (0, 2)),
    )

    logger.info("--- Broadcasting demo (3 worked examples) ---")
    broadcasting_demo()

    logger.info("--- normalize_broadcast: before/after demonstration ---")
    dataset = np.array([
        [10.0, 200.0, 1.0],
        [20.0, 250.0, 2.0],
        [30.0, 300.0, 3.0],
        [40.0, 350.0, 4.0],
    ])
    logger.info("Before (raw dataset):\n%s", dataset)
    logger.info("Column means before: %s", dataset.mean(axis=0))
    logger.info("Column stds before:  %s", dataset.std(axis=0))

    normalized = normalize_broadcast(dataset)
    logger.info("After (normalized):\n%s", normalized)
    logger.info("Column means after (should be ~0): %s", normalized.mean(axis=0))
    logger.info("Column stds after (should be ~1):  %s", normalized.std(axis=0))


if __name__ == "__main__":
    main()
