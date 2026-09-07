"""Unit tests for Day 1 Python Revision & Advanced Constructs (day1_python_core.py)."""

import pytest
import logging
import sys
from pathlib import Path

# Add project root and module directory to sys.path
file_dir = Path(__file__).parent
project_root = file_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(file_dir) not in sys.path:
    sys.path.insert(0, str(file_dir))

try:
    from epic1_numpy.day1_python_core import (
        build_model_config,
        aggregate_metrics,
        parse_dataset_sources,
        fibonacci_generator,
        flatten_nested_list,
        word_frequency,
        chunk_list,
        timer,
        log_call,
    )
except ImportError:
    from day1_python_core import (
        build_model_config,
        aggregate_metrics,
        parse_dataset_sources,
        fibonacci_generator,
        flatten_nested_list,
        word_frequency,
        chunk_list,
        timer,
        log_call,
    )



# --- 1. Utility Functions Tests ---

def test_build_model_config():
    config = build_model_config("conv2d", "relu", "dense", lr=0.01, dropout=0.2)
    assert config["layers"] == ["conv2d", "relu", "dense"]
    assert config["hyperparameters"] == {"lr": 0.01, "dropout": 0.2}
    assert config["layer_count"] == 3


def test_aggregate_metrics():
    res = aggregate_metrics(0.2, 0.4, accuracy=0.9, val_loss=0.1)
    assert res["positional_losses"] == [0.2, 0.4]
    assert res["named_metrics"] == {"accuracy": 0.9, "val_loss": 0.1}
    assert res["total_sum"] == pytest.approx(1.6)
    assert res["mean"] == pytest.approx(0.4)


def test_parse_dataset_sources():
    res = parse_dataset_sources("data1.csv", "data2.csv", batch_size=32, shuffle=True)
    assert res["sources"] == ["data1.csv", "data2.csv"]
    assert res["source_count"] == 2
    assert res["options"] == {"batch_size": 32, "shuffle": True}


# --- 2. Fibonacci Generator Tests ---

def test_fibonacci_generator_normal():
    result = list(fibonacci_generator(10))
    assert result == [0, 1, 1, 2, 3, 5, 8]


def test_fibonacci_generator_edge_zero():
    result = list(fibonacci_generator(0))
    assert result == [0]


def test_fibonacci_generator_edge_negative():
    result = list(fibonacci_generator(-5))
    assert result == []


# --- 3. Flatten Nested List Tests ---

def test_flatten_nested_list_standard():
    nested = [1, [2, [3, 4], 5], 6]
    assert flatten_nested_list(nested) == [1, 2, 3, 4, 5, 6]


def test_flatten_nested_list_edge_empty_and_single():
    assert flatten_nested_list([]) == []
    assert flatten_nested_list([42]) == [42]
    assert flatten_nested_list([[], [[]]]) == []


# --- 4. Word Frequency Tests ---

def test_word_frequency_standard():
    text = "Hello world! HELLO python, world."
    freq = word_frequency(text)
    assert freq == {"hello": 2, "world": 2, "python": 1}


def test_word_frequency_edge_empty():
    assert word_frequency("") == {}
    assert word_frequency("    ") == {}


# --- 5. Chunk List Tests ---

def test_chunk_list_standard():
    data = [1, 2, 3, 4, 5, 6, 7]
    chunks = list(chunk_list(data, size=3))
    assert chunks == [[1, 2, 3], [4, 5, 6], [7]]


def test_chunk_list_edge_cases():
    # Single element list
    assert list(chunk_list([10], size=5)) == [[10]]
    # Empty list
    assert list(chunk_list([], size=3)) == []
    # Chunk size larger than data
    assert list(chunk_list([1, 2], size=10)) == [[1, 2]]


def test_chunk_list_invalid_size():
    with pytest.raises(ValueError, match="Chunk size must be a positive integer"):
        list(chunk_list([1, 2, 3], size=0))


# --- 6. Decorator Tests ---

def test_decorators_metadata_and_execution(caplog):
    @timer
    @log_call
    def sample_func(a: int, b: int) -> int:
        """Sample function docstring."""
        return a + b

    with caplog.at_level(logging.INFO):
        res = sample_func(3, 4)

    assert res == 7
    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "Sample function docstring."
    assert "Calling function 'sample_func(3, 4)'" in caplog.text
    assert "Executed 'sample_func'" in caplog.text
