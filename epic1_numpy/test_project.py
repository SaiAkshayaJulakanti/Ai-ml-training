"""Pytest suite for project_student_analytics.py.

Run with: pytest -v
Coverage: pytest --cov=epic1_numpy.project_student_analytics --cov=common.numpy_utils
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

file_dir = Path(__file__).parent
project_root = file_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from epic1_numpy.project_student_analytics import (
        compute_subject_stats,
        correlation_between_subjects,
        generate_dataset,
        identify_at_risk_students,
        normalize_scores,
        rank_students,
        run_pipeline,
        save_dataset,
        save_report,
    )
except ImportError:
    from project_student_analytics import (
        compute_subject_stats,
        correlation_between_subjects,
        generate_dataset,
        identify_at_risk_students,
        normalize_scores,
        rank_students,
        run_pipeline,
        save_dataset,
        save_report,
    )

from common.numpy_utils import array_stats, rank_by_score, zscore_normalize


# --------------------------------------------------------------------------
# A known, hand-computable small dataset used throughout these tests.
# 4 students x 3 subjects, chosen so statistics can be verified by hand.
# --------------------------------------------------------------------------
KNOWN_SCORES = np.array([
    [90.0, 80.0, 70.0],   # student 0: avg 80
    [60.0, 50.0, 40.0],   # student 1: avg 50
    [100.0, 100.0, 100.0],  # student 2: avg 100
    [30.0, 20.0, 10.0],   # student 3: avg 20
])


# --------------------------------------------------------------------------
# generate_dataset
# --------------------------------------------------------------------------
def test_generate_dataset_shape():
    scores = generate_dataset(num_students=200, num_subjects=5)
    assert scores.shape == (200, 5)


def test_generate_dataset_is_reproducible_with_seed():
    """Same seed must produce the exact same dataset - required for reproducibility."""
    scores_a = generate_dataset(num_students=50, num_subjects=5, seed=42)
    scores_b = generate_dataset(num_students=50, num_subjects=5, seed=42)
    assert np.array_equal(scores_a, scores_b)


def test_generate_dataset_within_valid_range():
    scores = generate_dataset(num_students=200, num_subjects=5)
    assert scores.min() >= 0
    assert scores.max() <= 100


def test_save_dataset_writes_readable_csv(tmp_path):
    """The raw dataset must actually be saved and readable back, not just discarded."""
    scores = generate_dataset(num_students=10, num_subjects=5)
    filepath = str(tmp_path / "dataset.csv")
    save_dataset(scores, filepath)

    assert os.path.exists(filepath)
    loaded = np.loadtxt(filepath, delimiter=",", skiprows=1)
    assert loaded.shape == (10, 5)
    assert np.allclose(loaded, scores, atol=0.01)  # atol for the %.2f rounding on write


# --------------------------------------------------------------------------
# compute_subject_stats - correctness on the known small dataset
# --------------------------------------------------------------------------
def test_compute_subject_stats_known_values():
    stats = compute_subject_stats(KNOWN_SCORES)
    # Column 0 (first subject): [90, 60, 100, 30] -> mean = 70
    first_subject = list(stats.keys())[0]
    assert np.isclose(stats[first_subject]["mean"], 70.0)
    assert np.isclose(stats[first_subject]["min"], 30.0)
    assert np.isclose(stats[first_subject]["max"], 100.0)


def test_compute_subject_stats_returns_all_subjects():
    stats = compute_subject_stats(KNOWN_SCORES)
    assert len(stats) == KNOWN_SCORES.shape[1]
    for subject_stats in stats.values():
        assert set(subject_stats.keys()) == {"mean", "median", "std", "min", "max"}


# --------------------------------------------------------------------------
# rank_students - correctness on the known small dataset
# --------------------------------------------------------------------------
def test_rank_students_known_order():
    """Student averages are [80, 50, 100, 20] -> expected rank order: 2 (best), 0, 1, 3 (worst)."""
    ranks = rank_students(KNOWN_SCORES)
    # student 2 has the highest average (100) -> rank 0
    assert ranks[2] == 0
    # student 3 has the lowest average (20) -> rank 3 (last, since 4 students)
    assert ranks[3] == 3
    # student 0 (avg 80) should rank better than student 1 (avg 50)
    assert ranks[0] < ranks[1]


def test_rank_students_uses_all_ranks_exactly_once():
    ranks = rank_students(KNOWN_SCORES)
    assert sorted(ranks.tolist()) == list(range(len(KNOWN_SCORES)))


# --------------------------------------------------------------------------
# identify_at_risk_students - known threshold
# --------------------------------------------------------------------------
def test_identify_at_risk_students_known_threshold():
    """With threshold=55, students with avg [80, 50, 100, 20] -> students 1 and 3 are at risk."""
    at_risk = identify_at_risk_students(KNOWN_SCORES, threshold=55.0)
    assert set(at_risk.tolist()) == {1, 3}


def test_identify_at_risk_students_no_students_at_risk():
    """Edge case: a threshold below every average should flag nobody."""
    at_risk = identify_at_risk_students(KNOWN_SCORES, threshold=0.0)
    assert len(at_risk) == 0


def test_identify_at_risk_students_all_students_at_risk():
    """Edge case: a threshold above every average should flag everybody."""
    at_risk = identify_at_risk_students(KNOWN_SCORES, threshold=1000.0)
    assert len(at_risk) == len(KNOWN_SCORES)


# --------------------------------------------------------------------------
# normalize_scores
# --------------------------------------------------------------------------
def test_normalize_scores_mean_and_std():
    normalized = normalize_scores(KNOWN_SCORES.astype(float))
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-8)


def test_normalize_scores_preserves_shape():
    normalized = normalize_scores(KNOWN_SCORES.astype(float))
    assert normalized.shape == KNOWN_SCORES.shape


# --------------------------------------------------------------------------
# correlation_between_subjects - shape and symmetry
# --------------------------------------------------------------------------
def test_correlation_matrix_shape():
    corr = correlation_between_subjects(KNOWN_SCORES)
    n_subjects = KNOWN_SCORES.shape[1]
    assert corr.shape == (n_subjects, n_subjects)


def test_correlation_matrix_is_symmetric():
    corr = correlation_between_subjects(KNOWN_SCORES)
    assert np.allclose(corr, corr.T)


def test_correlation_matrix_diagonal_is_one():
    """A subject always correlates perfectly with itself."""
    corr = correlation_between_subjects(KNOWN_SCORES)
    assert np.allclose(np.diag(corr), 1.0)


# --------------------------------------------------------------------------
# save_report - both .txt and .json output
# --------------------------------------------------------------------------
def test_save_report_writes_text_file(tmp_path):
    report_data = {
        "num_students": 4,
        "num_subjects": 3,
        "subject_stats": compute_subject_stats(KNOWN_SCORES),
        "top_5": [(2, 100.0), (0, 80.0)],
        "at_risk_threshold": 55.0,
        "num_at_risk": 2,
        "at_risk_student_ids": [1, 3],
        "correlation_matrix": correlation_between_subjects(KNOWN_SCORES).tolist(),
    }
    filepath = str(tmp_path / "test_report.txt")
    save_report(report_data, filepath)

    assert os.path.exists(filepath)
    with open(filepath) as f:
        content = f.read()
    assert "STUDENT PERFORMANCE ANALYTICS REPORT" in content
    assert "At-Risk" in content


def test_save_report_writes_json_file(tmp_path):
    report_data = {
        "num_students": 4,
        "num_subjects": 3,
        "subject_stats": compute_subject_stats(KNOWN_SCORES),
        "at_risk_threshold": 55.0,
        "num_at_risk": 2,
        "at_risk_student_ids": [1, 3],
        "correlation_matrix": correlation_between_subjects(KNOWN_SCORES).tolist(),
    }
    filepath = str(tmp_path / "test_report.json")
    save_report(report_data, filepath)

    assert os.path.exists(filepath)
    import json
    with open(filepath) as f:
        loaded = json.load(f)
    assert loaded["num_students"] == 4
    assert loaded["at_risk_student_ids"] == [1, 3]


# --------------------------------------------------------------------------
# Full pipeline - end-to-end
# --------------------------------------------------------------------------
def test_run_pipeline_end_to_end(tmp_path):
    """The full pipeline must run without error and produce a real report file."""
    filepath = str(tmp_path / "pipeline_report.txt")
    report_data = run_pipeline(report_filepath=filepath)

    assert os.path.exists(filepath)
    assert report_data["num_students"] == 200
    assert report_data["num_subjects"] == 5
    assert len(report_data["subject_stats"]) == 5


# --------------------------------------------------------------------------
# common/numpy_utils.py - the shared, refactored utilities
# --------------------------------------------------------------------------
def test_common_array_stats_matches_numpy():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = array_stats(arr)
    assert np.isclose(stats["mean"], np.mean(arr))
    assert np.isclose(stats["std"], np.std(arr))


def test_common_zscore_normalize_rejects_zero_std_column():
    """Edge case: a column of identical values must raise, not divide by zero."""
    arr = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    with pytest.raises(ValueError):
        zscore_normalize(arr, axis=0)


def test_common_rank_by_score_descending():
    scores = np.array([10.0, 30.0, 20.0])
    ranks = rank_by_score(scores, descending=True)
    assert ranks[1] == 0   # 30 is highest -> rank 0
    assert ranks[0] == 2   # 10 is lowest -> rank 2
