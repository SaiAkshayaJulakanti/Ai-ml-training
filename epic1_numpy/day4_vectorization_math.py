"""Day 4: Vectorization & Mathematical Operations

This module replaces loop-based logic with vectorized NumPy operations,
benchmarks the difference, and covers core linear algebra and statistical
functions used throughout ML pipelines.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 0. @timer decorator - imported from Day 1, per the task spec
# --------------------------------------------------------------------------
try:
    from epic1_numpy.day1_python_core import timer
except ImportError:
    from day1_python_core import timer


# --------------------------------------------------------------------------
# 1. Loop-based vs vectorized: sum of squares
# --------------------------------------------------------------------------
@timer
def sum_of_squares_loop(arr: np.ndarray) -> float:
    """Compute sum of squares using an explicit Python loop (the slow way)."""
    total = 0.0
    for x in arr:
        total += x * x
    return total


@timer
def sum_of_squares_vectorized(arr: np.ndarray) -> float:
    """Compute sum of squares using vectorized NumPy operations (the fast way).

    `arr * arr` squares every element in compiled C code in one pass;
    `.sum()` reduces it to a scalar - no Python-level loop at all.
    """
    return float(np.sum(arr * arr))


# --------------------------------------------------------------------------
# 2. Loop-based vs vectorized: dot product
# --------------------------------------------------------------------------
@timer
def dot_product_loop(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product of two 1D arrays using an explicit loop."""
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


@timer
def dot_product_vectorized(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product using np.dot (vectorized, compiled)."""
    return float(np.dot(a, b))


# --------------------------------------------------------------------------
# 3. Loop-based vs vectorized: elementwise distance
# --------------------------------------------------------------------------
@timer
def elementwise_distance_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute |a[i] - b[i]| for every position, using an explicit loop."""
    result = np.empty(len(a))
    for i in range(len(a)):
        result[i] = abs(a[i] - b[i])
    return result


@timer
def elementwise_distance_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute |a - b| elementwise using vectorized NumPy operations."""
    return np.abs(a - b)


# --------------------------------------------------------------------------
# 4. Benchmarking loop vs vectorized
# --------------------------------------------------------------------------
def benchmark_loop_vs_vectorized(n: int) -> Dict[str, Dict[str, float]]:
    """Benchmark loop-based vs vectorized implementations on n-element arrays.

    Each implementation is run once as an untimed "warmup" before the
    timed run. This matters because the very first NumPy operation on a
    freshly allocated array pays a one-time cost (memory allocation,
    page faults) that has nothing to do with the algorithm's real speed -
    without a warmup, that one-time cost can make a fast vectorized
    function look artificially slow on its first call.

    Args:
        n: Size of the arrays to benchmark with (e.g. 1_000_000).

    Returns:
        A dict keyed by operation name, each containing 'loop_time',
        'vectorized_time', and 'speedup' (loop_time / vectorized_time).
    """
    rng = np.random.default_rng(seed=42)
    a = rng.random(n)
    b = rng.random(n)

    def _time_once(func: Callable[..., Any], *args: Any) -> float:
        func(*args)  # warmup - untimed, not the same call being measured
        start = time.perf_counter()
        func(*args)
        return time.perf_counter() - start

    results: Dict[str, Dict[str, float]] = {}

    operations = [
        ("sum_of_squares", sum_of_squares_loop.__wrapped__, sum_of_squares_vectorized.__wrapped__, (a,)),
        ("dot_product", dot_product_loop.__wrapped__, dot_product_vectorized.__wrapped__, (a, b)),
        ("elementwise_distance", elementwise_distance_loop.__wrapped__, elementwise_distance_vectorized.__wrapped__, (a, b)),
    ]

    for name, loop_fn, vec_fn, args in operations:
        loop_time = _time_once(loop_fn, *args)
        vec_time = _time_once(vec_fn, *args)
        results[name] = {
            "loop_time": loop_time,
            "vectorized_time": vec_time,
            "speedup": loop_time / vec_time if vec_time > 0 else float("inf"),
        }

    return results


# --------------------------------------------------------------------------
# 5. Matrix operations
# --------------------------------------------------------------------------
def matrix_ops_demo() -> None:
    """Demonstrate np.dot, the @ operator, transpose, determinant, and inverse."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])

    logger.info("np.dot(a, b):\n%s", np.dot(a, b))
    logger.info("a @ b (equivalent to np.dot for 2D arrays):\n%s", a @ b)
    logger.info("a.T (transpose):\n%s", a.T)
    logger.info("np.linalg.det(a): %s", np.linalg.det(a))
    logger.info("np.linalg.inv(a):\n%s", np.linalg.inv(a))


def matrix_ops_report(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Build a report of matrix operations between `a` and `b`.

    Args:
        a: A 2D array.
        b: A 2D array with a compatible shape for matrix multiplication
            with `a` (a.shape[1] == b.shape[0]).

    Returns:
        A dict with keys:
            - 'product': a @ b, or None if shapes are incompatible
            - 'transpose_a': a.T
            - 'transpose_b': b.T
            - 'det_a': determinant of `a` if square, else None
            - 'det_b': determinant of `b` if square, else None
            - 'inverse_a': inverse of `a` if square and invertible, else None
            - 'inverse_b': inverse of `b` if square and invertible, else None
            - 'errors': a list of human-readable strings describing any
              operations that could not be performed and why
    """
    report: Dict[str, Any] = {
        "product": None,
        "transpose_a": a.T,
        "transpose_b": b.T,
        "det_a": None,
        "det_b": None,
        "inverse_a": None,
        "inverse_b": None,
        "errors": [],
    }

    # Matrix product - requires a.shape[1] == b.shape[0]
    try:
        report["product"] = a @ b
    except ValueError as exc:
        report["errors"].append(f"product: shapes {a.shape} and {b.shape} incompatible ({exc})")

    # Determinant - only defined for square matrices
    for name, mat in (("a", a), ("b", b)):
        if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
            report[f"det_{name}"] = float(np.linalg.det(mat))
        else:
            report["errors"].append(f"det_{name}: matrix with shape {mat.shape} is not square")

    # Inverse - only defined for square, non-singular matrices
    for name, mat in (("a", a), ("b", b)):
        if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
            try:
                report[f"inverse_{name}"] = np.linalg.inv(mat)
            except np.linalg.LinAlgError as exc:
                report["errors"].append(f"inverse_{name}: matrix is not invertible ({exc})")
        else:
            report["errors"].append(f"inverse_{name}: matrix with shape {mat.shape} is not square")

    return report


# --------------------------------------------------------------------------
# 6. Statistical functions
# --------------------------------------------------------------------------
def statistics_demo(arr: np.ndarray) -> Dict[str, Any]:
    """Demonstrate mean, median, std, var, and percentile along specified axes.

    Args:
        arr: A 2D array. Statistics are computed both overall (no axis)
            and per-column (axis=0), to show how the `axis` argument
            changes the shape of the result.

    Returns:
        A dict of the computed statistics.
    """
    stats = {
        "mean_overall": float(np.mean(arr)),
        "mean_per_column": np.mean(arr, axis=0),
        "median_overall": float(np.median(arr)),
        "median_per_column": np.median(arr, axis=0),
        "std_overall": float(np.std(arr)),
        "std_per_column": np.std(arr, axis=0),
        "var_overall": float(np.var(arr)),
        "var_per_column": np.var(arr, axis=0),
        "percentile_25_overall": float(np.percentile(arr, 25)),
        "percentile_75_overall": float(np.percentile(arr, 75)),
    }
    return stats


# --------------------------------------------------------------------------
# 7. Practical exercise: fully vectorized euclidean distance matrix
# --------------------------------------------------------------------------
def euclidean_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the pairwise Euclidean distance matrix between rows of `a` and `b`.

    Fully vectorized - no Python loops. For `a` of shape (m, d) and `b` of
    shape (n, d), returns a matrix of shape (m, n) where element [i, j] is
    the Euclidean distance between a[i] and b[j].

    How this avoids loops: `a[:, np.newaxis, :]` reshapes `a` to
    (m, 1, d) and broadcasts against `b` of shape (n, d) -> (1, n, d),
    producing all pairwise differences at once as a (m, n, d) array.
    Squaring, summing over the last axis, and taking the square root
    then collapses that to the (m, n) distance matrix.

    Args:
        a: Array of shape (m, d).
        b: Array of shape (n, d).

    Returns:
        Distance matrix of shape (m, n).

    Raises:
        ValueError: If `a` and `b` don't share the same number of columns
            (the same dimensionality `d`).
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"euclidean_distance_matrix requires 2D arrays, got ndim={a.ndim} and ndim={b.ndim}"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"a and b must have the same number of columns (dimensionality); "
            f"got a.shape={a.shape}, b.shape={b.shape}"
        )

    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]   # shape (m, n, d)
    squared_dist = np.sum(diff ** 2, axis=-1)           # shape (m, n)
    return np.sqrt(squared_dist)


# --------------------------------------------------------------------------
# 8. Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Day 4: Vectorization & Mathematical Operations ===")

    logger.info("--- Loop vs vectorized correctness check (small array) ---")
    small = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    small_b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    logger.info(
        "sum_of_squares: loop=%s vectorized=%s",
        sum_of_squares_loop(small), sum_of_squares_vectorized(small),
    )
    logger.info(
        "dot_product: loop=%s vectorized=%s",
        dot_product_loop(small, small_b), dot_product_vectorized(small, small_b),
    )
    logger.info(
        "elementwise_distance: loop=%s vectorized=%s",
        elementwise_distance_loop(small, small_b), elementwise_distance_vectorized(small, small_b),
    )

    logger.info("--- Benchmark: loop vs vectorized on 1,000,000-element arrays ---")
    results = benchmark_loop_vs_vectorized(1_000_000)
    logger.info("%-22s | %-12s | %-16s | %-10s", "Operation", "Loop (s)", "Vectorized (s)", "Speedup")
    logger.info("-" * 70)
    for op_name, timings in results.items():
        logger.info(
            "%-22s | %-12.6f | %-16.6f | %-10.1fx",
            op_name, timings["loop_time"], timings["vectorized_time"], timings["speedup"],
        )

    logger.info("--- Matrix operations demo ---")
    matrix_ops_demo()

    logger.info("--- matrix_ops_report demo (square, invertible matrices) ---")
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    report = matrix_ops_report(a, b)
    logger.info("Product:\n%s", report["product"])
    logger.info("det_a=%s, det_b=%s", report["det_a"], report["det_b"])
    logger.info("Errors: %s", report["errors"])

    logger.info("--- matrix_ops_report demo (non-invertible / non-square edge case) ---")
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])   # determinant is 0 -> not invertible
    non_square = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    report2 = matrix_ops_report(singular, non_square)
    logger.info("Errors for singular/non-square inputs: %s", report2["errors"])

    logger.info("--- Statistical functions demo ---")
    dataset = np.array([
        [10.0, 200.0],
        [20.0, 250.0],
        [30.0, 300.0],
        [40.0, 350.0],
    ])
    stats = statistics_demo(dataset)
    for key, value in stats.items():
        logger.info("%s: %s", key, value)

    logger.info("--- euclidean_distance_matrix demo ---")
    points_a = np.array([[0.0, 0.0], [1.0, 1.0]])
    points_b = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    logger.info("Distance matrix:\n%s", euclidean_distance_matrix(points_a, points_b))


if __name__ == "__main__":
    main()
