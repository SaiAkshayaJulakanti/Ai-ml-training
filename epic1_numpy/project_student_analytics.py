"""epic1_numpy/project_student_analytics.py - Student Performance Analytics Engine

Epic 1 capstone project: an end-to-end NumPy-only pipeline (no pandas)
that generates a synthetic student score dataset, computes statistics,
ranks students, flags at-risk students, normalizes scores, and computes
inter-subject correlation - consolidating everything from Days 1-4.

Run directly: python project_student_analytics.py
Produces both console output and a written report file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Make the shared common/ package importable both when this file is run
# as a script from inside epic1_numpy/, and when imported as part of the
# ai-ml-training package (same fallback pattern used in Days 1-4 tests).
file_dir = Path(__file__).parent
project_root = file_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.numpy_utils import (  # noqa: E402
    array_stats,
    rank_by_score,
    timer,
    write_json_report,
    write_text_report,
    zscore_normalize,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration (no hardcoded magic numbers scattered through the pipeline -
# every tunable value lives here, named, in one place)
# --------------------------------------------------------------------------
NUM_STUDENTS = 200
SUBJECT_NAMES = ["Math", "Science", "English", "History", "ComputerScience"]
NUM_SUBJECTS = len(SUBJECT_NAMES)
RANDOM_SEED = 42
SCORE_MEAN = 70.0
SCORE_STD = 15.0
SCORE_MIN = 0
SCORE_MAX = 100
AT_RISK_THRESHOLD = 60.0
DEFAULT_REPORT_FILEPATH = "student_analytics_report.txt"
DEFAULT_DATASET_FILEPATH = "student_scores_dataset.csv"


# --------------------------------------------------------------------------
# 1. Synthetic dataset generation
# --------------------------------------------------------------------------
def generate_dataset(
    num_students: int = NUM_STUDENTS,
    num_subjects: int = NUM_SUBJECTS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate a synthetic (num_students x num_subjects) score matrix.

    Scores are drawn from a normal distribution (mean=SCORE_MEAN,
    std=SCORE_STD) and clipped to [SCORE_MIN, SCORE_MAX] so they behave
    like realistic 0-100 exam scores. Seeded via np.random.seed for
    reproducibility, per the task spec.

    Args:
        num_students: Number of student rows to generate.
        num_subjects: Number of subject columns to generate.
        seed: Random seed for reproducibility.

    Returns:
        A (num_students, num_subjects) float array of scores.
    """
    np.random.seed(seed)
    scores = np.random.normal(loc=SCORE_MEAN, scale=SCORE_STD, size=(num_students, num_subjects))
    return np.clip(scores, SCORE_MIN, SCORE_MAX)


# --------------------------------------------------------------------------
# 2. Per-subject statistics
# --------------------------------------------------------------------------
def compute_subject_stats(scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute mean, median, std, min, max for each subject (column).

    Args:
        scores: A (num_students, num_subjects) score matrix.

    Returns:
        A dict keyed by subject name, each value a dict of statistics.
    """
    stats_by_axis = array_stats(scores, axis=0)  # each value: array of length num_subjects

    result: Dict[str, Dict[str, float]] = {}
    for i, subject in enumerate(SUBJECT_NAMES[: scores.shape[1]]):
        result[subject] = {
            "mean": float(stats_by_axis["mean"][i]),
            "median": float(stats_by_axis["median"][i]),
            "std": float(stats_by_axis["std"][i]),
            "min": float(stats_by_axis["min"][i]),
            "max": float(stats_by_axis["max"][i]),
        }
    return result


# --------------------------------------------------------------------------
# 3. Ranking students
# --------------------------------------------------------------------------
def rank_students(scores: np.ndarray) -> np.ndarray:
    """Rank students by their average score across all subjects, using np.argsort.

    Args:
        scores: A (num_students, num_subjects) score matrix.

    Returns:
        A 1D int array of length num_students, where result[i] is
        student i's rank (0 = highest average score).
    """
    average_scores = scores.mean(axis=1)
    return rank_by_score(average_scores, descending=True)


# --------------------------------------------------------------------------
# 4. At-risk student detection (boolean masking)
# --------------------------------------------------------------------------
def identify_at_risk_students(scores: np.ndarray, threshold: float = AT_RISK_THRESHOLD) -> np.ndarray:
    """Identify students whose average score falls below `threshold`.

    Uses boolean masking: a student is "at risk" if their average score
    across all subjects is below the threshold.

    Args:
        scores: A (num_students, num_subjects) score matrix.
        threshold: The average-score cutoff below which a student is
            flagged as at-risk.

    Returns:
        A 1D int array of student indices (row numbers) who are at risk,
        in ascending index order.
    """
    average_scores = scores.mean(axis=1)
    mask = average_scores < threshold
    return np.where(mask)[0]


# --------------------------------------------------------------------------
# 5. Normalization (z-score, via the shared broadcasting utility)
# --------------------------------------------------------------------------
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Z-score normalize scores per-subject (column), using broadcasting only.

    Thin wrapper around common.numpy_utils.zscore_normalize - this is
    the actual refactored logic originally built in Day 3's
    normalize_broadcast.
    """
    return zscore_normalize(scores, axis=0)


# --------------------------------------------------------------------------
# 6. Correlation between subjects
# --------------------------------------------------------------------------
def correlation_between_subjects(scores: np.ndarray) -> np.ndarray:
    """Compute the subject-to-subject correlation matrix using np.corrcoef.

    Args:
        scores: A (num_students, num_subjects) score matrix.

    Returns:
        A (num_subjects, num_subjects) correlation matrix. Diagonal
        entries are always 1.0 (a subject perfectly correlates with
        itself); the matrix is symmetric.
    """
    # np.corrcoef expects variables as ROWS by default, so transpose:
    # each subject (column in `scores`) becomes a row for corrcoef.
    return np.corrcoef(scores, rowvar=False)


# --------------------------------------------------------------------------
# 6b. Raw dataset export - so the underlying data is actually inspectable,
# not just the derived summary statistics in the report
# --------------------------------------------------------------------------
def save_dataset(scores: np.ndarray, filepath: str) -> None:
    """Write the raw student x subject score matrix to a CSV file.

    Uses np.savetxt (not pandas, per the project constraint), with a
    header row naming each subject column so the file is self-describing
    on its own.

    Args:
        scores: The (num_students, num_subjects) raw score matrix.
        filepath: Output CSV path.
    """
    header = ",".join(SUBJECT_NAMES[: scores.shape[1]])
    np.savetxt(filepath, scores, delimiter=",", header=header, comments="", fmt="%.2f")


# --------------------------------------------------------------------------
# 7. Report generation
# --------------------------------------------------------------------------
def save_report(report_data: Dict[str, Any], filepath: str) -> None:
    """Write the full analytics report to a file (.txt or .json, based on extension).

    Args:
        report_data: The full report dict (stats, rankings, at-risk list,
            correlation matrix, etc).
        filepath: Output path. A '.json' extension writes structured
            JSON; anything else writes a formatted plain-text report.
    """
    if filepath.endswith(".json"):
        write_json_report(report_data, filepath)
        return

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("STUDENT PERFORMANCE ANALYTICS REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Dataset: {report_data['num_students']} students x "
                 f"{report_data['num_subjects']} subjects")
    lines.append("")

    lines.append("--- Per-Subject Statistics ---")
    for subject, stats in report_data["subject_stats"].items():
        lines.append(
            f"{subject:<18} mean={stats['mean']:6.2f}  median={stats['median']:6.2f}  "
            f"std={stats['std']:6.2f}  min={stats['min']:6.2f}  max={stats['max']:6.2f}"
        )
    lines.append("")

    lines.append("--- Top 5 Ranked Students (by average score) ---")
    for student_id, avg_score in report_data["top_5"]:
        lines.append(f"  Student #{student_id:<4} average score: {avg_score:.2f}")
    lines.append("")

    lines.append(f"--- At-Risk Students (average < {report_data['at_risk_threshold']}) ---")
    lines.append(f"  Count: {report_data['num_at_risk']}")
    lines.append(f"  Student IDs: {report_data['at_risk_student_ids']}")
    lines.append("")

    lines.append("--- Subject Correlation Matrix ---")
    subjects = list(report_data["subject_stats"].keys())
    header = " " * 18 + "".join(f"{s[:10]:>12}" for s in subjects)
    lines.append(header)
    corr = np.array(report_data["correlation_matrix"])
    for i, subject in enumerate(subjects):
        row = "".join(f"{corr[i, j]:>12.3f}" for j in range(len(subjects)))
        lines.append(f"{subject:<18}{row}")
    lines.append("")

    lines.append("=" * 60)

    write_text_report(lines, filepath)


# --------------------------------------------------------------------------
# 8. Full pipeline
# --------------------------------------------------------------------------
@timer
def run_pipeline(
    report_filepath: str = DEFAULT_REPORT_FILEPATH,
    dataset_filepath: str = DEFAULT_DATASET_FILEPATH,
) -> Dict[str, Any]:
    """Run the full analytics pipeline end-to-end and return the report data.

    Args:
        report_filepath: Where to write the generated summary report.
        dataset_filepath: Where to write the raw generated score dataset
            (so the underlying data is directly inspectable, not just
            the derived statistics in the report).

    Returns:
        The full report data dict (same content written to the report).
    """
    scores = generate_dataset()
    save_dataset(scores, dataset_filepath)

    subject_stats = compute_subject_stats(scores)
    ranks = rank_students(scores)
    at_risk_ids = identify_at_risk_students(scores)
    normalized = normalize_scores(scores)
    correlation = correlation_between_subjects(scores)

    average_scores = scores.mean(axis=1)
    top_5_indices = np.argsort(average_scores)[::-1][:5]
    top_5 = [(int(idx), float(average_scores[idx])) for idx in top_5_indices]

    report_data: Dict[str, Any] = {
        "num_students": scores.shape[0],
        "num_subjects": scores.shape[1],
        "subject_stats": subject_stats,
        "top_5": top_5,
        "at_risk_threshold": AT_RISK_THRESHOLD,
        "num_at_risk": len(at_risk_ids),
        "at_risk_student_ids": at_risk_ids.tolist(),
        "correlation_matrix": correlation.tolist(),
        "normalized_scores_sample": normalized[:3].tolist(),  # first 3 rows, for spot-checking
        "ranks_sample": ranks[:10].tolist(),
    }

    save_report(report_data, report_filepath)
    return report_data


# --------------------------------------------------------------------------
# 9. CLI entry point
# --------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Student Performance Analytics Engine ===")

    report_data = run_pipeline()

    logger.info("--- Per-Subject Statistics ---")
    for subject, stats in report_data["subject_stats"].items():
        logger.info(
            "%-18s mean=%6.2f  median=%6.2f  std=%6.2f  min=%6.2f  max=%6.2f",
            subject, stats["mean"], stats["median"], stats["std"], stats["min"], stats["max"],
        )

    logger.info("--- Top 5 Ranked Students ---")
    for student_id, avg_score in report_data["top_5"]:
        logger.info("  Student #%d: average score %.2f", student_id, avg_score)

    logger.info(
        "--- At-Risk Students (average < %s): %d students ---",
        report_data["at_risk_threshold"], report_data["num_at_risk"],
    )
    logger.info("  IDs: %s", report_data["at_risk_student_ids"])

    logger.info("--- Subject Correlation Matrix ---")
    logger.info("\n%s", np.array(report_data["correlation_matrix"]))

    logger.info("Dataset written to: %s", DEFAULT_DATASET_FILEPATH)
    logger.info("Report written to: %s", DEFAULT_REPORT_FILEPATH)


if __name__ == "__main__":
    main()
