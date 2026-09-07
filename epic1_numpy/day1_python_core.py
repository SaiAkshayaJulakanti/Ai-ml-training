"""Day 1: Python Revision + Advanced Python for AI/ML

This module contains core Python utilities, generator functions, and decorators
designed with an AI/ML lens.
"""

from collections import Counter
from functools import wraps
import logging
import re
import time
from typing import Any, Callable, Generator, Iterator, List, Dict, TypeVar, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# --- Decorators ---

def timer(func: F) -> F:
    """Decorator that logs the execution time of a function."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logger.info("Executed '%s' in %.6f seconds", func.__name__, elapsed_time)
        return result
    return wrapper  # type: ignore[return-value]


def log_call(func: F) -> F:
    """Decorator that logs the function name and arguments before execution."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.info("Calling function '%s(%s)'", func.__name__, signature)
        return func(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


# --- Utility Functions (*args, **kwargs) ---

def build_model_config(*args: str, **kwargs: Any) -> Dict[str, Any]:
    """Assembles a model configuration dictionary from positional architecture layers

    and keyword hyperparameter options.

    Args:
        *args: Names of model layers in execution order (e.g., 'conv2d', 'relu').
        **kwargs: Model hyperparameters (e.g., learning_rate=0.001, batch_size=32).

    Returns:
        Dict[str, Any]: Consolidated configuration containing 'layers' list and hyperparameters.
    """
    config: Dict[str, Any] = {
        "layers": list(args),
        "hyperparameters": kwargs,
        "layer_count": len(args)
    }
    return config


def aggregate_metrics(*args: Union[int, float], **kwargs: Union[int, float]) -> Dict[str, Any]:
    """Aggregates unnamed numerical loss values and named metric values.

    Args:
        *args: Unnamed epoch loss values or metric numbers.
        **kwargs: Named metrics (e.g., accuracy=0.95, f1_score=0.92).

    Returns:
        Dict[str, Any]: Aggregated dictionary with computed averages and combined metrics.
    """
    all_numeric = [v for v in args] + [v for v in kwargs.values() if isinstance(v, (int, float))]
    total_val = sum(all_numeric) if all_numeric else 0.0
    mean_val = total_val / len(all_numeric) if all_numeric else 0.0

    return {
        "positional_losses": list(args),
        "named_metrics": kwargs,
        "total_sum": total_val,
        "mean": round(mean_val, 4)
    }


def parse_dataset_sources(*args: str, **kwargs: Any) -> Dict[str, Any]:
    """Processes multiple dataset file source paths and ingestion parameters.

    Args:
        *args: Paths to dataset files.
        **kwargs: Data loading options (e.g., shuffle=True, format="csv").

    Returns:
        Dict[str, Any]: Dataset ingestion schema representation.
    """
    return {
        "sources": list(args),
        "source_count": len(args),
        "options": kwargs
    }


# --- Generator Function ---

def fibonacci_generator(n: int) -> Generator[int, None, None]:
    """Lazily yields Fibonacci numbers up to maximum threshold N (value <= N).

    Args:
        n (int): Upper bound value limit for generated Fibonacci numbers.

    Yields:
        int: Next Fibonacci number in sequence <= N.
    """
    if n < 0:
        return

    a, b = 0, 1
    while a <= n:
        yield a
        a, b = b, a + b


# --- Practical Coding Exercises ---

def flatten_nested_list(data: List[Any]) -> List[Any]:
    """Recursively flattens an arbitrarily nested list using recursion + comprehension.

    Args:
        data (List[Any]): Potentially nested list.

    Returns:
        List[Any]: Flattened single-level list.
    """
    return [
        item
        for element in data
        for item in (flatten_nested_list(element) if isinstance(element, list) else [element])
    ]


def word_frequency(text: str) -> Dict[str, int]:
    """Calculates word frequency count in a text string using collections.Counter.

    Args:
        text (str): Input text string.

    Returns:
        Dict[str, int]: Mapping of normalized words to their frequency counts.
    """
    if not text or not text.strip():
        return {}
    
    words = re.findall(r"\b\w+\b", text.lower())
    return dict(Counter(words))


@timer
@log_call
def chunk_list(data: List[Any], size: int) -> Generator[List[Any], None, None]:
    """Generator yielding sublists (chunks) of specified size from input data.

    Stacked with @timer and @log_call decorators.

    Args:
        data (List[Any]): Input data list to split into chunks.
        size (int): Maximum size of each chunk.

    Yields:
        List[Any]: Sublist chunk.
    """
    if size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    for i in range(0, len(data), size):
        yield data[i : i + size]


if __name__ == "__main__":
    logger.info("=== Demonstrating Day 1 Functions ===")

    # 1. Model config utility
    config = build_model_config("conv2d", "maxpool", "flatten", "dense", learning_rate=0.001, epochs=10)
    logger.info("Model Config: %s", config)

    # 2. Metrics utility
    metrics = aggregate_metrics(0.5, 0.4, 0.25, accuracy=0.94, val_loss=0.22)
    logger.info("Aggregated Metrics: %s", metrics)

    # 3. Dataset sources utility
    sources = parse_dataset_sources("data/train.csv", "data/val.csv", shuffle=True, batch_size=64)
    logger.info("Dataset Sources: %s", sources)

    # 4. Fibonacci generator
    fib_nums = list(fibonacci_generator(20))
    logger.info("Fibonacci numbers <= 20: %s", fib_nums)

    # 5. Flatten nested list
    nested = [1, [2, [3, 4], 5], [6, [7, 8]], 9]
    flattened = flatten_nested_list(nested)
    logger.info("Flattened list: %s", flattened)

    # 6. Word frequency
    sample_text = "AI for ML and ML for AI: building intelligence with Python!"
    freq = word_frequency(sample_text)
    logger.info("Word Frequency: %s", freq)

    # 7. Stacked decorators demonstration on chunk_list
    dataset_sample = list(range(1, 11))
    logger.info("Demonstrating stacked @timer and @log_call on chunk_list:")
    generator_obj = chunk_list(dataset_sample, size=3)
    chunks = list(generator_obj)
    logger.info("Generated Chunks: %s", chunks)
