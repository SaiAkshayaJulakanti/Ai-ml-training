"""Day 2: NumPy Arrays - Creation & Core Operations

This module covers NumPy array creation methods, attribute inspection,
reshaping semantics, concatenation, and dtype casting - the foundation
for all numerical computing that follows.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Array creation methods
# --------------------------------------------------------------------------
def create_from_list() -> np.ndarray:
    """Create an array directly from a nested Python list (2D)."""
    return np.array([[1, 2, 3], [4, 5, 6]])


def create_zeros(shape: Tuple[int, ...]) -> np.ndarray:
    """Create an array of the given shape filled with zeros."""
    return np.zeros(shape)


def create_ones(shape: Tuple[int, ...]) -> np.ndarray:
    """Create an array of the given shape filled with ones."""
    return np.ones(shape)


def create_arange(start: int, stop: int, step: int = 1) -> np.ndarray:
    """Create a 1D array of evenly spaced values within [start, stop) using a step size."""
    return np.arange(start, stop, step)


def create_linspace(start: float, stop: float, num: int = 50) -> np.ndarray:
    """Create a 1D array of `num` evenly spaced values between start and stop (inclusive)."""
    return np.linspace(start, stop, num)


def create_identity(n: int) -> np.ndarray:
    """Create an n x n identity matrix using np.eye."""
    return np.eye(n)


def create_random(rows: int, cols: int) -> np.ndarray:
    """Create a rows x cols array of random floats in [0, 1) using np.random.rand."""
    return np.random.rand(rows, cols)


# --------------------------------------------------------------------------
# 2. Array inspector utility
# --------------------------------------------------------------------------
def inspect_array(arr: np.ndarray) -> Dict[str, Any]:
    """Inspect an array's core attributes and print them as a formatted table.

    Reports .shape, .ndim, .dtype, .size, and .itemsize (bytes per element),
    plus the total memory footprint (size * itemsize).

    Args:
        arr: The NumPy array to inspect.

    Returns:
        A dict containing the same attributes, for programmatic use.
    """
    info = {
        "shape": arr.shape,
        "ndim": arr.ndim,
        "dtype": arr.dtype,
        "size": arr.size,
        "itemsize": arr.itemsize,
        "total_bytes": arr.size * arr.itemsize,
    }

    logger.info("=" * 46)
    logger.info("%-15s | %s", "Attribute", "Value")
    logger.info("-" * 46)
    for key, value in info.items():
        logger.info("%-15s | %s", key, value)
    logger.info("=" * 46)

    return info


# --------------------------------------------------------------------------
# 3. Reshaping: .reshape() vs .flatten() vs .ravel()
# --------------------------------------------------------------------------
def reshape_pipeline(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape `arr` into `shape`, raising a clear ValueError on incompatible shapes.

    Notes on reshape/flatten/ravel (see inline comments below for details):
        - .reshape(shape): returns a new VIEW (when possible) with the given
          shape; total element count must match the original, or NumPy raises
          ValueError.
        - .flatten(): always returns a 1D COPY of the array - safe to mutate
          without affecting the original.
        - .ravel(): returns a 1D VIEW when possible (falls back to a copy only
          if the data isn't contiguous) - faster than flatten() but mutating
          the result CAN mutate the original array.

    Args:
        arr: Input array to reshape.
        shape: Target shape as a tuple.

    Returns:
        The reshaped array.

    Raises:
        ValueError: If `shape` is not compatible with `arr`'s total element count.
    """
    try:
        return arr.reshape(shape)
    except ValueError as exc:
        raise ValueError(
            f"Cannot reshape array of size {arr.size} into shape {shape}: {exc}"
        ) from exc


def flatten_vs_ravel_demo(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Demonstrate .flatten() vs .ravel() and their copy/view distinction.

    - flatten() returns a COPY: modifying it never touches `arr`.
    - ravel() returns a VIEW when possible: modifying it CAN touch `arr`.
    """
    flat_copy = arr.flatten()
    flat_view = arr.ravel()

    # Prove the distinction: mutate each result and check if `arr` changed.
    original_first_value = arr.flat[0]

    flat_copy[0] = -999
    copy_affects_original = bool(arr.flat[0] != original_first_value)

    flat_view[0] = -999
    view_affects_original = bool(arr.flat[0] == -999)

    return {
        "flatten_result": flat_copy,
        "ravel_result": flat_view,
        # Cast to plain Python bool: NumPy comparisons return np.bool_,
        # which is NOT identical to Python's True/False under `is` checks
        # even though it is equal under ==. This bit us in testing below.
        "flatten_is_copy": bool(not copy_affects_original),
        "ravel_is_view": view_affects_original,
    }


# --------------------------------------------------------------------------
# 4. Concatenation / stacking
# --------------------------------------------------------------------------
def stack_arrays(arr_list: List[np.ndarray], axis: int = 0) -> np.ndarray:
    """Concatenate a list of arrays along the given axis using np.concatenate.

    Args:
        arr_list: List of arrays to concatenate. Must have matching shapes
            along every axis except `axis`.
        axis: The axis along which to concatenate.

    Returns:
        The concatenated array.

    Raises:
        ValueError: If shapes are incompatible for concatenation along `axis`.
    """
    return np.concatenate(arr_list, axis=axis)


def vstack_demo(arr_list: List[np.ndarray]) -> np.ndarray:
    """Stack arrays vertically (row-wise) using np.vstack."""
    return np.vstack(arr_list)


def hstack_demo(arr_list: List[np.ndarray]) -> np.ndarray:
    """Stack arrays horizontally (column-wise) using np.hstack."""
    return np.hstack(arr_list)


# --------------------------------------------------------------------------
# 5. Practical coding exercises
# --------------------------------------------------------------------------
def create_identity_matrix(n: int) -> np.ndarray:
    """Return an n x n identity matrix.

    Args:
        n: Size of the (square) identity matrix.

    Returns:
        An n x n identity matrix as a NumPy array.
    """
    return np.eye(n)


def random_matrix_stats(rows: int, cols: int) -> Dict[str, float]:
    """Generate a random matrix and return its min/max/mean/std statistics.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        Dict with keys 'min', 'max', 'mean', 'std'.
    """
    matrix = np.random.rand(rows, cols)
    return {
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "mean": float(matrix.mean()),
        "std": float(matrix.std()),
    }


def dtype_cast_demo(arr: np.ndarray, target_dtype: str) -> np.ndarray:
    """Cast `arr` to `target_dtype` and log the memory impact.

    Casting int32 -> float64 doubles the per-element byte size (4 -> 8 bytes),
    so a large array's memory footprint doubles. Casting float64 -> int32
    truncates decimals (no rounding) and halves memory, at the cost of
    precision loss.
    """
    original_bytes = arr.nbytes
    casted = arr.astype(target_dtype)
    new_bytes = casted.nbytes

    logger.info(
        "dtype cast: %s (%d bytes) -> %s (%d bytes) | delta: %+d bytes",
        arr.dtype, original_bytes, casted.dtype, new_bytes, new_bytes - original_bytes,
    )
    return casted


# --------------------------------------------------------------------------
# 6. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 2: NumPy Array Creation & Core Operations ===")
    logger.info("NumPy version: %s", np.__version__)

    # --- Array creation ---
    logger.info("--- Array creation ---")
    logger.info("from_list:\n%s", create_from_list())
    logger.info("zeros(2,3):\n%s", create_zeros((2, 3)))
    logger.info("ones(3,2):\n%s", create_ones((3, 2)))
    logger.info("arange(0,10,2): %s", create_arange(0, 10, 2))
    logger.info("linspace(0,1,5): %s", create_linspace(0, 1, 5))
    logger.info("eye(4):\n%s", create_identity(4))
    logger.info("random(2,2):\n%s", create_random(2, 2))

    # --- Inspector on 1D/2D/3D arrays ---
    logger.info("--- Array inspector demo ---")
    arr_1d = np.arange(10)
    arr_2d = np.ones((3, 4))
    arr_3d = np.zeros((2, 3, 4))
    for label, arr in [("1D", arr_1d), ("2D", arr_2d), ("3D", arr_3d)]:
        logger.info("Inspecting %s array:", label)
        inspect_array(arr)

    # --- Reshape / flatten / ravel ---
    logger.info("--- Reshape pipeline ---")
    base = np.arange(12)
    reshaped = reshape_pipeline(base, (3, 4))
    logger.info("reshape(3,4):\n%s", reshaped)

    try:
        reshape_pipeline(base, (5, 5))
    except ValueError as e:
        logger.info("Expected error on invalid reshape: %s", e)

    logger.info("--- flatten vs ravel demo ---")
    demo_arr = np.arange(6).reshape(2, 3)
    result = flatten_vs_ravel_demo(demo_arr.copy())
    logger.info(
        "flatten_is_copy=%s, ravel_is_view=%s",
        result["flatten_is_copy"], result["ravel_is_view"],
    )

    # --- Concatenation ---
    logger.info("--- Stacking / concatenation ---")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    logger.info("concatenate axis=0:\n%s", stack_arrays([a, b], axis=0))
    logger.info("vstack:\n%s", vstack_demo([a, b]))
    logger.info("hstack:\n%s", hstack_demo([a, b]))

    # --- Practical exercises ---
    logger.info("--- Practical exercises ---")
    logger.info("create_identity_matrix(5):\n%s", create_identity_matrix(5))
    logger.info("random_matrix_stats(4,4): %s", random_matrix_stats(4, 4))

    # --- dtype casting demo ---
    logger.info("--- dtype casting demo ---")
    int_arr = np.ones((3, 3), dtype=np.int32)
    dtype_cast_demo(int_arr, "float64")

    # --- Memory usage comparison table across dtypes for a 1000x1000 array ---
    logger.info("--- Memory usage comparison (1000x1000 array) ---")
    dtypes = ["int8", "int32", "int64", "float32", "float64"]
    logger.info("%-10s | %-15s", "dtype", "total memory (MB)")
    logger.info("-" * 30)
    for dt in dtypes:
        sample = np.ones((1000, 1000), dtype=dt)
        mb = sample.nbytes / (1024 * 1024)
        logger.info("%-10s | %-15.4f", dt, mb)


if __name__ == "__main__":
    main()
