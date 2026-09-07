# 🚀 AI/ML Training — Day 1: Python Revision & Advanced Constructs

Welcome to the **AI/ML Training Repository**! This project provides foundational Python utilities, advanced constructs (decorators, generators, comprehensions, `*args`/`**kwargs`), and an automated test suite designed with an Artificial Intelligence and Machine Learning lens.

---

## 📌 1. Objective & Overview

Before building complex Neural Networks or Data Engineering pipelines, high performance in core Python is essential. This repository focuses on:
- **Memory Efficiency**: Using **Generators** to lazily stream data without overloading RAM.
- **Function Monitoring**: Writing reusable **Decorators** (`@timer` and `@log_call`) to log function execution time and call signatures.
- **Dynamic Arguments**: Flexible configuration using `*args` and `**kwargs`.
- **Data Manipulation**: Recursive list comprehension for flattening nested lists and `collections.Counter` for NLP word frequency counts.
- **Robust Quality**: 100% test coverage using **Pytest** for both standard operations and edge cases.

---

## 📁 2. Repository Layout

```text
ai-ml-training/
├── .gitignore               # Excludes virtual environments (venv_aiml/), pycache, and build artifacts
├── README.md                # Comprehensive project documentation
├── common/                  # Shared helper tools across epics
│   ├── .gitkeep
│   └── __init__.py
├── epic1_numpy/             # Epic 1: NumPy & Python Core Foundations
│   ├── __init__.py
│   ├── day1_python_core.py  # All 7 utility/generator functions & 2 decorators
│   └── test_day1.py         # 14 Pytest automated test cases
└── epic2_pandas/            # Epic 2: Data Manipulation with Pandas (Upcoming)
    ├── .gitkeep
    └── __init__.py
```

---

## 🛠️ 3. Features & Functions Breakdown

### ⏱️ Decorators (Function Wrappers)
- **`@timer`**: Measures function execution duration using high-precision `time.perf_counter()` and logs the elapsed time cleanly using Python's standard `logging` module.
- **`@log_call`**: Logs the exact function name, positional arguments (`*args`), and keyword arguments (`**kwargs`) before execution.

### 🧰 Utility Functions (`*args`, `**kwargs`)
- **`build_model_config(*args, **kwargs)`**: Accepts sequential architecture layer names in `*args` (e.g., `'conv2d'`, `'dense'`) and hyperparameters in `**kwargs` (e.g., `learning_rate=0.001`), assembling a model blueprint dictionary.
- **`aggregate_metrics(*args, **kwargs)`**: Accepts unnamed loss values and named metric scores (e.g., `accuracy=0.95`), computing total sums and rounded averages.
- **`parse_dataset_sources(*args, **kwargs)`**: Processes dataset file paths and data loading options (e.g., `batch_size=64`, `shuffle=True`).

### ⚡ Generator Function
- **`fibonacci_generator(n: int)`**: Lazily yields Fibonacci numbers $\le N$. Unlike standard lists that allocate all values in memory simultaneously, this generator computes numbers on-the-fly to conserve RAM.

### 🏋️ Practical Exercises
- **`flatten_nested_list(data: list)`**: Flattens arbitrarily nested lists (e.g. `[1, [2, [3, 4], 5], 6]` $\rightarrow$ `[1, 2, 3, 4, 5, 6]`) using recursion combined with list comprehension.
- **`word_frequency(text: str)`**: Normalizes text (lowercase, punctuation removal) and counts word frequency using `collections.Counter`.
- **`chunk_list(data: list, size: int)`**: Generator yielding sublists/batches of size `size`. Wrapped with stacked `@timer` and `@log_call` decorators to demonstrate stacked execution output.

---

## 🖥️ 4. How to Run the Project

### Step 1: Open Terminal & Navigate to Project Root
```bash
cd "c:\Users\HAI\OneDrive\Desktop\Vibe Coding Assignment\ai-ml-training"
```

### Step 2: Run the Main Demonstration Script
Executes `day1_python_core.py` and displays logging outputs for all functions and stacked decorators:
```bash
python epic1_numpy/day1_python_core.py
```

---

## 🧪 5. Running the Pytest Suite

The project includes **14 automated unit tests** in `epic1_numpy/test_day1.py` covering standard cases and edge cases ($N=0$ Fibonacci, empty lists `[]`, single element lists `[42]`, empty text strings `""`, invalid chunk sizes).

To run the tests with verbose output:
```bash
pytest -v epic1_numpy/test_day1.py
```
*(Or alternatively: `python -m pytest -v epic1_numpy/test_day1.py`)*

---

## 🔀 6. Git Workflow & Commit Requirements

To commit your progress and push to your GitHub repository:

```bash
# Initialize git repository (if not already done)
git init

# Stage all files
git add .

# Create commit with required Day 1 commit message
git commit -m "Day1: Python core + advanced constructs (decorators, generators, comprehensions)"

# Set main branch and push
git branch -M main
git push -u origin main
```

---

## 💡 Summary Table of Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Core Functions** | [`day1_python_core.py`](file:///c:/Users/HAI/OneDrive/Desktop/Vibe%20Coding%20Assignment/ai-ml-training/epic1_numpy/day1_python_core.py) | 7 working functions, 2 decorators, 1 generator with full type hints |
| **Unit Tests** | [`test_day1.py`](file:///c:/Users/HAI/OneDrive/Desktop/Vibe%20Coding%20Assignment/ai-ml-training/epic1_numpy/test_day1.py) | 14 Pytest unit tests (100% passing) |
| **Git Exclusion** | [`.gitignore`](file:///c:/Users/HAI/OneDrive/Desktop/Vibe%20Coding%20Assignment/ai-ml-training/.gitignore) | Excludes `venv_aiml/`, `__pycache__`, `.pytest_cache` |
| **Documentation** | [`README.md`](file:///c:/Users/HAI/OneDrive/Desktop/Vibe%20Coding%20Assignment/ai-ml-training/README.md) | Comprehensive, simple user guide & technical documentation |
