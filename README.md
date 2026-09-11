# 🚀 AI/ML Training — Epic 1: Python & NumPy Foundations

Welcome to the **AI/ML Training Repository**! This project covers a full week
(Epic 1) of foundational work: core Python, advanced constructs, and NumPy —
from basic array creation all the way to a complete end-to-end analytics
project — all building toward the numerical computing skills used
throughout AI/ML pipelines.

**Status: Epic 1 Complete** ✅ — tagged `epic1-complete`

---

## 📌 1. Objective & Overview

This repository is organized day-by-day across one training week:

| Day | Topic | File |
|---|---|---|
| Day 1 | Python revision + advanced constructs (decorators, generators, comprehensions) | `epic1_numpy/day1_python_core.py` |
| Day 2 | NumPy array creation, dtypes, reshaping | `epic1_numpy/day2_numpy_basics.py` |
| Day 3 | Indexing, slicing, boolean/fancy indexing, broadcasting | `epic1_numpy/day3_indexing_broadcasting.py` |
| Day 4 | Vectorization, linear algebra, performance benchmarking | `epic1_numpy/day4_vectorization_math.py` |
| Day 5 | Capstone project — Student Performance Analytics Engine | `epic1_numpy/project_student_analytics.py` |

Each day builds on the previous one — Day 5 in particular consolidates and
**reuses** logic from Days 1–4 via a shared `common/numpy_utils.py` module,
rather than duplicating it.

---

## 📁 2. Repository Layout

```text
ai-ml-training/
├── .gitignore                        # Excludes venv_aiml/, pycache, build artifacts
├── requirements.txt                  # numpy, pytest, pytest-cov
├── README.md                         # This file
├── common/                           # Shared helper tools across epics
│   ├── __init__.py
│   └── numpy_utils.py                # Refactored shared logic (timer, stats,
│                                      # z-score normalize, ranking, report I/O)
├── epic1_numpy/                      # Epic 1: Python & NumPy Foundations
│   ├── __init__.py
│   ├── day1_python_core.py           # Day 1: decorators, generators, *args/**kwargs
│   ├── test_day1.py                  # 14 tests
│   ├── day2_numpy_basics.py          # Day 2: array creation, dtypes, reshaping
│   ├── test_day2.py                  # 18 tests
│   ├── day3_indexing_broadcasting.py # Day 3: masking, fancy indexing, broadcasting
│   ├── test_day3.py                  # 20 tests
│   ├── day4_vectorization_math.py    # Day 4: vectorization, linalg, benchmarking
│   ├── test_day4.py                  # 13 tests
│   ├── project_student_analytics.py  # Day 5: capstone project (see below)
│   ├── test_project.py               # 22 tests
│   ├── student_scores_dataset.csv    # Generated: raw 200x5 synthetic dataset
│   └── student_analytics_report.txt  # Generated: full analytics report
└── epic2_pandas/                     # Epic 2: Data Manipulation with Pandas (Upcoming)
    ├── .gitkeep
    └── __init__.py
```

---

## 🛠️ 3. Features & Functions Breakdown

### Day 1 — Python Core & Advanced Constructs
- **`@timer`** / **`@log_call`** decorators for execution timing and call
  logging.
- **`fibonacci_generator`** — lazy generator, yields Fibonacci numbers up
  to N without pre-building a list.
- **`*args`/`**kwargs` utilities**: `build_model_config`,
  `aggregate_metrics`, `parse_dataset_sources`.
- **`flatten_nested_list`**, **`word_frequency`**, **`chunk_list`** —
  recursion, `collections.Counter`, and a generator-based chunker with
  stacked `@timer` + `@log_call` decorators.

### Day 2 — NumPy Array Creation & Core Operations
- Array creation: `np.zeros`, `np.ones`, `np.arange`, `np.linspace`,
  `np.eye`, `np.random.rand`.
- **`inspect_array()`** — formatted inspector reporting shape, ndim,
  dtype, size, and memory footprint for 1D/2D/3D arrays.
- **`.reshape()` vs `.flatten()` vs `.ravel()`** — documented view-vs-copy
  semantics.
- **`create_identity_matrix`**, **`random_matrix_stats`**,
  **`reshape_pipeline`**, dtype casting with memory-impact comparison.

### Day 3 — Indexing, Slicing & Broadcasting
- Basic slicing (1D/2D: rows, columns, sub-matrices) — always a **view**.
- Boolean masking and **`filter_outliers`** — always a flat **copy**.
- Fancy indexing and **`select_rows_by_index`** — always a **copy**.
- **`normalize_broadcast`** — z-score normalization via broadcasting,
  zero explicit loops (now delegates to `common.numpy_utils.zscore_normalize`).
- Broadcasting rules documented inline with 3 worked examples (scalar+array,
  1D+2D, and a mismatched-shape failure case).

### Day 4 — Vectorization & Mathematical Operations
- Loop vs. vectorized implementations of sum-of-squares, dot product, and
  elementwise distance, each timed with Day 1's **`@timer`** decorator
  (imported directly, not duplicated).
- **`benchmark_loop_vs_vectorized`** — measured 40x–218x real speedups on
  1,000,000-element arrays (with a proper warmup run to avoid cold-start
  timing artifacts).
- **`matrix_ops_report`** — product, transpose, determinant, inverse,
  gracefully handling non-square and non-invertible (`LinAlgError`) matrices.
- **`euclidean_distance_matrix`** — fully vectorized pairwise distances,
  zero loops.
- Statistical functions (`mean`, `median`, `std`, `var`, `percentile`)
  demonstrated along specified axes.

### Day 5 — Capstone: Student Performance Analytics Engine
An end-to-end, NumPy-only (no pandas) pipeline simulating a real
numerical-analysis workflow:
- Generates a reproducible synthetic dataset (200 students × 5 subjects,
  `np.random.seed(42)`).
- **`compute_subject_stats`** — mean/median/std/min/max per subject.
- **`rank_students`** — ranks by average score using `np.argsort`.
- **`identify_at_risk_students`** — boolean-mask-based threshold detection.
- **`normalize_scores`** — column-wise z-score normalization via
  broadcasting.
- **`correlation_between_subjects`** — subject-to-subject correlation
  matrix via `np.corrcoef`.
- **`save_dataset`** / **`save_report`** — writes the raw dataset to
  `student_scores_dataset.csv` and a full formatted report to
  `student_analytics_report.txt` (or `.json`).
- Reuses shared logic from `common/numpy_utils.py` instead of
  re-implementing Day 1–4 patterns — genuine refactoring, not duplication.

### `common/numpy_utils.py` — Shared Utilities
Extracted, reusable logic pulled out of the week's daily work so it has a
single source of truth:
- `timer` — re-exported from Day 1's decorator (not a duplicate copy).
- `array_stats` — the mean/median/std/min/max pattern from Days 2 & 4.
- `zscore_normalize` — the broadcasting-based normalization from Day 3,
  now used by both Day 3 and the Day 5 project.
- `rank_by_score` — ranking via `np.argsort`.
- `write_text_report` / `write_json_report` — generic report-writing
  helpers.

---

## 🖥️ 4. How to Run the Project

### Step 1: Open Terminal & Navigate to Project Root
```bash
cd "c:\Users\HAI\OneDrive\Desktop\Vibe Coding Assignment\ai-ml-training"
```

### Step 2: Create the Virtual Environment (first-time setup only)
```bash
python -m venv venv_aiml
```

### Step 3: Activate the Virtual Environment
```bash
venv_aiml\Scripts\activate
```
*(macOS/Linux: `source venv_aiml/bin/activate`)*

If PowerShell blocks activation with an execution-policy error, run this
once (as your normal user, not admin):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run Any Day's Demonstration Script
```bash
python epic1_numpy/day1_python_core.py
python epic1_numpy/day2_numpy_basics.py
python epic1_numpy/day3_indexing_broadcasting.py
python epic1_numpy/day4_vectorization_math.py
```

### Step 6: Run the Capstone Project
```bash
python epic1_numpy/project_student_analytics.py
```
This prints a full console report **and** writes two files into
`epic1_numpy/`:
- `student_scores_dataset.csv` — the raw generated dataset
- `student_analytics_report.txt` — the full analytics report

---

## 🧪 5. Running the Test Suite

The project includes **87 automated pytest cases** across all five days
(14 + 18 + 20 + 13 + 22), all passing.

Run everything at once:
```bash
pytest -v epic1_numpy/
```

Or an individual day:
```bash
pytest -v epic1_numpy/test_day1.py
pytest -v epic1_numpy/test_project.py
```

### Coverage (Day 5 project)
```bash
pytest --cov=epic1_numpy.project_student_analytics --cov=common.numpy_utils -v epic1_numpy/test_project.py
```
Core Day 5 files measured at **86%+ coverage**.

---

## 🔀 6. Git Workflow & Commit History

Each day was committed separately with its own descriptive message:

```bash
git add .
git commit -m "Day1: Python core + advanced constructs (decorators, generators, comprehensions)"
git commit -m "Day2: NumPy array creation, reshaping, dtype handling"
git commit -m "Day3: Indexing, boolean masking, fancy indexing, broadcasting"
git commit -m "Day4: Vectorization, linear algebra ops, performance benchmarking"
git commit -m "Day5: Epic1 deliverable - Student Analytics Engine (NumPy end-to-end project)"
git push
```

Epic 1 completion is marked with an annotated tag:
```bash
git tag epic1-complete
git push --tags
```

---

## 💡 7. Summary Table of Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Day 1 Core** | `epic1_numpy/day1_python_core.py` | 7 functions, 2 decorators, 1 generator |
| **Day 2 NumPy Basics** | `epic1_numpy/day2_numpy_basics.py` | Array creation, inspector utility, reshape semantics |
| **Day 3 Indexing/Broadcasting** | `epic1_numpy/day3_indexing_broadcasting.py` | Masking, fancy indexing, broadcasting rules |
| **Day 4 Vectorization** | `epic1_numpy/day4_vectorization_math.py` | Loop vs. vectorized benchmarks, linear algebra |
| **Day 5 Capstone Project** | `epic1_numpy/project_student_analytics.py` | End-to-end NumPy analytics pipeline |
| **Shared Utilities** | `common/numpy_utils.py` | Refactored, reusable logic from Days 1–4 |
| **Generated Dataset** | `epic1_numpy/student_scores_dataset.csv` | Raw 200×5 synthetic student scores |
| **Generated Report** | `epic1_numpy/student_analytics_report.txt` | Full computed analytics report |
| **Test Suites** | `epic1_numpy/test_*.py` | 87 passing pytest cases across all 5 days |
| **Git Exclusion** | `.gitignore` | Excludes `venv_aiml/`, `__pycache__`, `.pytest_cache` |
| **Dependencies** | `requirements.txt` | Pinned minimum versions: numpy, pytest, pytest-cov |
| **Documentation** | `README.md` | This file |
