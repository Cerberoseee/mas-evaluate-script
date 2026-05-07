"""Project-wide defaults."""

from __future__ import annotations

from pathlib import Path


DEFAULT_DATASET_NAME = "SWE-bench/SWE-bench_Lite"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_TASK_COUNT = 25
DEFAULT_REPEATS = 1
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_TOTAL_TOKEN_BUDGET = 250_000
DEFAULT_OUTPUT_ROOT = Path(".")
DEFAULT_MANIFEST_PATH = Path("manifests/swebench_lite_25.json")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_REPO_CACHE_DIR = Path(".cache/repos")
DEFAULT_GRADE_WORK_DIR = Path(".cache/grades")

ARM_MINI_SWE_AGENT = "mini_swe_agent"
ARM_MAS_CENTRALIZE = "mas_centralize"
ARM_MAS_DECENTRALIZED = "mas_decentralized"

DEFAULT_ARMS = [
    ARM_MAS_CENTRALIZE,
    ARM_MAS_DECENTRALIZED,
]

DEFAULT_REPO_ORDER = [
    "astropy/astropy",
    "django/django",
    "marshmallow-code/marshmallow",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/astroid",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sqlfluff/sqlfluff",
    "sphinx-doc/sphinx",
    "sympy/sympy",
]
