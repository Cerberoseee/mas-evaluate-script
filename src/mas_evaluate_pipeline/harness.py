"""Workspace preparation and grading helpers."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from .models import FailureClass, GradeResult, HarnessConfig, PreparedTask, RunRecord, TaskInstance, utc_now

logger = logging.getLogger(__name__)

# ── Provision configuration ────────────────────────────────────────────────────
#
# The strategy follows the official SWE-bench harness (swebench/harness/constants/python.py):
#
#   1. Create a uv-managed venv with the correct Python version.
#   2. Install any pre-install packages (build deps or pinned runtime deps).
#   3. For repos that support it, run `pip install -e .[test]` (editable install).
#      This compiles C/Cython extensions and pulls in the full declared test stack —
#      the same steps the official Docker-based harness performs.
#   4. For repos that do NOT support editable install, fall back to installing a
#      common baseline (pytest, hypothesis, packaging) plus repo-specific extras.
#   5. Write any file stubs needed for repos that still can't be editably installed.

# Baseline packages for repos without an editable install (step 4 above).
_COMMON_PROVISION_PACKAGES: list[str] = [
    "pytest",
    "hypothesis",
    "packaging",
]

# Repos that support `pip install -e .[test]` (or equivalent).
# Key   → "org/repo"
# Value → pip argument list appended after `pip install`  (e.g. ["-e", ".[test]"])
#
# Mirrors the "install" field in SWE-bench constants/python.py SPECS_*.
# When a repo is listed here its _REPO_PROVISION_EXTRAS are installed FIRST
# (build deps), then the editable install runs. Common packages are skipped
# because the editable install pulls in the full test stack from the repo's
# own setup.cfg / pyproject.toml.
_REPO_EDITABLE_INSTALL: dict[str, list[str]] = {
    # astropy v5.x: compiles _compiler.so, _parse_times.so, erfa, etc.
    # Pulls in pytest-astropy, pytest-doctestplus, pytest-xdist, etc.
    # --no-build-isolation: use the venv's pinned setuptools<67.3 during the
    # build instead of letting uv pull setuptools 82.x into a fresh isolated env
    # (which would break `from setuptools.dep_util import newer_group` in
    # astropy/wcs/setup_package.py).
    "astropy/astropy": ["-e", ".[test]", "--verbose", "--no-build-isolation"],
}

# Packages installed BEFORE the editable install (build deps for compiled repos)
# or instead of it (runtime extras for repos without editable install).
_REPO_PROVISION_EXTRAS: dict[str, list[str]] = {
    # Build deps needed so `pip install -e .[test]` can compile Cython extensions.
    # setuptools is pinned to <67.3 because astropy/wcs/setup_package.py at this
    # era uses `from setuptools.dep_util import newer_group`, which was removed in
    # setuptools 67.3.0.  The editable install runs with --no-build-isolation so it
    # uses the venv's pinned setuptools rather than pulling the latest in an
    # isolated build env.
    # numpy must be present before ``uv pip install -e . --no-build-isolation``: several
    # ``*/setup_package.py`` files import numpy during ``prepare_metadata_for_build_editable``.
    # hypothesis is also here because astropy's [test] extra omits it.
    "astropy/astropy": [
        "numpy",
        "cython",
        "setuptools<67.3",
        "wheel",
        "extension-helpers",
        "hypothesis",
    ],
    "django/django": ["sqlparse", "asgiref"],
    "matplotlib/matplotlib": [
        "pillow", "contourpy", "cycler", "fonttools",
        "kiwisolver", "pyparsing", "python-dateutil",
    ],
    "pallets/flask": ["click", "itsdangerous", "jinja2", "markupsafe", "werkzeug"],
    "psf/requests": ["certifi", "charset-normalizer", "idna", "urllib3"],
    "pytest-dev/pytest": ["pluggy", "iniconfig", "attrs"],
    "scikit-learn/scikit-learn": [
        "cython", "setuptools", "scipy", "joblib", "threadpoolctl",
    ],
    "sphinx-doc/sphinx": ["docutils", "jinja2", "pygments", "babel"],
    "sympy/sympy": [],
    "pydata/xarray": ["scipy", "pandas", "netCDF4"],
    "marshmallow-code/marshmallow": [],
    "mwaskom/seaborn": ["pandas", "scipy", "statsmodels"],
    "sqlfluff/sqlfluff": ["appdirs", "colorama", "tqdm", "toml"],
    "pylint-dev/pylint": ["astroid", "isort", "mccabe", "tomlkit"],
    "pylint-dev/astroid": [],
}

# File stubs written into the workspace for repos that cannot be editably
# installed (e.g. missing build toolchain). Stubs are added to .git/info/exclude
# so `git add -A` never commits them into the agent's patch.
# Repos using editable install do NOT need stubs — compiled extensions are real.
_REPO_VERSION_STUBS: dict[str, list[tuple[str, str]]] = {
    # No stubs needed for astropy when editable install is active.
}


def find_swebench_instance_report(run_dir: Path, record: RunRecord, instance_id: str) -> Path | None:
    """Return SWE-bench per-instance ``report.json`` if present."""
    run_dir = run_dir.resolve()
    primary = (
        run_dir
        / "logs"
        / "run_evaluation"
        / record.run_id
        / record.arm.value
        / instance_id
        / "report.json"
    )
    if primary.is_file():
        return primary
    for path in sorted(run_dir.rglob("report.json")):
        if path.is_file() and "run_evaluation" in path.parts:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(data, dict) and instance_id in data:
                return path
    return None


def summarize_swebench_instance_tests(report_payload: dict[str, Any], instance_id: str) -> dict[str, Any]:
    """Extract fail_to_pass / pass_to_pass mismatch lists from SWE-bench ``report.json``."""
    inst = report_payload.get(instance_id)
    if not isinstance(inst, dict):
        return {}
    ts = inst.get("tests_status")
    if not isinstance(ts, dict):
        return {}

    def bucket(name: str) -> tuple[list[str], list[str]]:
        raw = ts.get(name)
        if not isinstance(raw, dict):
            return [], []
        return list(raw.get("success") or []), list(raw.get("failure") or [])

    ftp_ok, ftp_bad = bucket("FAIL_TO_PASS")
    ptp_ok, ptp_bad = bucket("PASS_TO_PASS")
    _ptf_ok, ptf_bad = bucket("PASS_TO_FAIL")
    _ftf_ok, ftf_bad = bucket("FAIL_TO_FAIL")

    parts: list[str] = []
    if ftp_bad:
        parts.append(f"{len(ftp_bad)} fail_to_pass test(s) still failing (expected to pass)")
    if ptp_bad:
        parts.append(f"{len(ptp_bad)} pass_to_pass test(s) regressed (should still pass)")
    if ptf_bad:
        parts.append(f"{len(ptf_bad)} pass_to_fail regression(s)")
    if ftf_bad:
        parts.append(f"{len(ftf_bad)} unexpected fail_to_fail outcome(s)")
    summary = (
        "; ".join(parts)
        if parts
        else "See tests_status: no failures in FAIL_TO_PASS, PASS_TO_PASS, PASS_TO_FAIL, or FAIL_TO_FAIL buckets."
    )

    return {
        "summary": summary,
        "fail_to_pass_still_failing": ftp_bad,
        "fail_to_pass_now_passing": ftp_ok,
        "pass_to_pass_still_passing": ptp_ok,
        "pass_to_pass_regressions": ptp_bad,
        "pass_to_fail": ptf_bad,
        "fail_to_fail": ftf_bad,
        "tests_status": ts,
    }


def attach_swebench_grading_analysis(
    details: dict[str, Any],
    *,
    run_dir: Path,
    record: RunRecord,
    task: TaskInstance,
    resolved: bool,
) -> None:
    path = find_swebench_instance_report(run_dir, record, task.instance_id)
    if path is None:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    info = summarize_swebench_instance_tests(payload, task.instance_id)
    if not info:
        return
    info["report_json_path"] = str(path.resolve())
    if resolved and all(
        info.get(field) == []
        for field in (
            "fail_to_pass_still_failing",
            "pass_to_pass_regressions",
            "pass_to_fail",
            "fail_to_fail",
        )
    ):
        info["summary"] = "Marked resolved — no failing regressions or stuck fail_to_pass outcomes in report."
    details["swebench_grading"] = info


class WorkspacePreparer:
    """Hydrate per-run git workspaces from local mirrors or remote clones."""

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config
        self._repo_cache_dir = Path(config.repo_cache_dir).resolve()

    def prepare(
        self,
        task: TaskInstance,
        workspace_dir: Path,
        metadata_path: Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> PreparedTask:
        workspace_dir.parent.mkdir(parents=True, exist_ok=True)
        source_repo = self._ensure_source_repo(task, progress=progress)
        if workspace_dir.exists():
            if progress:
                progress(f"[prepare-workspace] removing existing workspace={workspace_dir}")
            shutil.rmtree(workspace_dir)
        if progress:
            progress(f"[prepare-workspace] cloning source_repo={source_repo} -> workspace={workspace_dir}")
        self._clone_workspace(source_repo, workspace_dir)
        if task.base_commit:
            if progress:
                progress(f"[prepare-workspace] checking out base_commit={task.base_commit}")
            self._run(["git", "checkout", task.base_commit], cwd=workspace_dir)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "instance_id": task.instance_id,
                    "repo": task.repo,
                    "base_commit": task.base_commit,
                    "problem_statement": task.problem_statement,
                    "hints_text": task.hints_text,
                    "fail_to_pass": task.fail_to_pass,
                    "pass_to_pass": task.pass_to_pass,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self._config.auto_provision_runtime:
            self._provision_runtime(workspace_dir, task, progress=progress)
        return PreparedTask(
            task=task,
            workspace_dir=str(workspace_dir),
            metadata_path=str(metadata_path),
            source_repo_dir=str(source_repo),
        )

    def _provision_runtime(
        self,
        workspace_dir: Path,
        task: TaskInstance,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Create workspace/venv/, install test deps, and optionally run editable install.

        Mirrors the official SWE-bench harness strategy:
          - Repos with compiled extensions: pre-install build deps, then
            `pip install -e .[test]` (compiles C/Cython, pulls in full test stack).
          - Repos without editable install: install common baseline + repo extras.

        Runs best-effort: failures are logged but do not abort the run.
        """
        workspace_dir = workspace_dir.resolve()
        python_version = (
            self._config.provision_python_versions.get(task.repo)
            or self._config.provision_python_version
        )
        venv_dir = workspace_dir / "venv"
        venv_python = venv_dir / "bin" / "python"
        prefix = f"[provision] repo={task.repo} python={python_version}"

        def _log(msg: str) -> None:
            logger.info("%s %s", prefix, msg)
            if progress:
                progress(f"{prefix} {msg}")

        def _uv_pip(*args: str) -> None:
            self._run_captured(
                ["uv", "pip", "install", "--python", str(venv_python)] + list(args),
                cwd=workspace_dir,
            )

        # ── Step 1: resolve Python and create venv ─────────────────────────────
        # Use `uv python find` to get the absolute path so uv never silently
        # falls back to the system Python when the requested version is available.
        python_executable: str | None = None
        try:
            _log(f"resolving Python {python_version} via uv")
            self._run_captured(["uv", "python", "install", python_version], cwd=workspace_dir)
            raw = self._run_captured(["uv", "python", "find", python_version], cwd=workspace_dir)
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            found = lines[-1] if lines else ""
            if found and found.startswith("/"):
                python_executable = found
                _log(f"resolved Python {python_version} → {python_executable}")
        except Exception as exc:  # noqa: BLE001
            _log(f"WARNING: could not resolve Python {python_version} via uv — {exc}; trying bare version")

        python_arg = python_executable or python_version
        try:
            _log(f"creating venv at {venv_dir} using {python_arg}")
            self._run_captured(
                ["uv", "venv", str(venv_dir), "--python", python_arg],
                cwd=workspace_dir,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"WARNING: venv creation failed — {exc}; agent will start without pre-provisioned env")
            return

        try:
            actual_ver = self._run_captured(
                [str(venv_python), "-c",
                 "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                cwd=workspace_dir,
            ).strip()
            expected_prefix = ".".join(python_version.split(".")[:2])
            if not actual_ver.startswith(expected_prefix):
                _log(f"WARNING: venv is Python {actual_ver}, expected {python_version}")
            else:
                _log(f"venv Python confirmed: {actual_ver}")
        except Exception as exc:  # noqa: BLE001
            _log(f"WARNING: could not verify venv Python — {exc}")

        editable_args = _REPO_EDITABLE_INSTALL.get(task.repo)

        # ── Step 2: pre-install packages ───────────────────────────────────────
        # For repos with editable install: just build deps (extras list).
        # For repos without: common baseline + extras + any user-supplied packages.
        extras = _REPO_PROVISION_EXTRAS.get(task.repo, [])
        if editable_args:
            if extras:
                try:
                    _log(f"installing build deps: {extras}")
                    _uv_pip(*extras)
                except Exception as exc:  # noqa: BLE001
                    _log(f"WARNING: build dep install failed — {exc}")
        else:
            packages = (
                list(_COMMON_PROVISION_PACKAGES)
                + list(self._config.provision_extra_packages)
                + extras
            )
            try:
                _log(f"installing packages: {packages}")
                _uv_pip(*packages)
            except Exception as exc:  # noqa: BLE001
                _log(f"WARNING: package install failed — {exc}")

        # ── Step 3: editable install (mirrors official harness `install` command) ──
        # uv venv does not create venv/bin/pip, so we use `uv pip install` which
        # handles editable installs and delegates to the build backend (setuptools/
        # Cython) to compile extensions into the venv's site-packages.
        if editable_args:
            _log(f"running editable install: uv pip install {' '.join(editable_args)}")
            try:
                _uv_pip(*editable_args)
                _log("editable install complete")
            except Exception as exc:  # noqa: BLE001
                _log(f"WARNING: editable install failed — {exc}; agent may lack compiled extensions")

        # ── Step 4: file stubs (fallback for repos without editable install) ───
        # With a successful editable install, C extensions are real compiled .so
        # files — no stubs needed. Stubs are still supported for repos that cannot
        # be editably installed.
        stubs = _REPO_VERSION_STUBS.get(task.repo, [])
        for rel_path, content in stubs:
            target = workspace_dir / rel_path
            try:
                target.write_text(content, encoding="utf-8")
                _log(f"wrote stub: {rel_path}")
            except Exception as exc:  # noqa: BLE001
                _log(f"WARNING: could not write stub {rel_path} — {exc}")

        # Exclude stubs from git so they never appear in the agent's patch.diff.
        if stubs:
            exclude_file = workspace_dir / ".git" / "info" / "exclude"
            try:
                existing = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
                new_entries = [p for p, _ in stubs if p not in existing]
                if new_entries:
                    with exclude_file.open("a", encoding="utf-8") as fh:
                        fh.write("\n# provision stubs – do not commit\n")
                        for p in new_entries:
                            fh.write(f"{p}\n")
                    _log(f"excluded stubs from git: {new_entries}")
            except Exception as exc:  # noqa: BLE001
                _log(f"WARNING: could not update .git/info/exclude — {exc}")

        _log("provision complete")

    def _ensure_source_repo(
        self,
        task: TaskInstance,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        mirror = self._config.repo_mirrors.get(task.repo)
        if mirror:
            if progress:
                progress(f"[prepare-source] using repo_mirror repo={task.repo} path={mirror}")
            return Path(mirror).resolve()
        self._repo_cache_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = self._repo_cache_dir / task.repo.replace("/", "__")
        legacy_repo_dir = self._repo_cache_dir / ".cache" / "repos" / task.repo.replace("/", "__")
        if not repo_dir.exists() and legacy_repo_dir.exists():
            if progress:
                progress(
                    f"[prepare-source] found legacy nested cache repo={task.repo} path={legacy_repo_dir}; "
                    "reusing it for this run"
                )
            repo_dir = legacy_repo_dir
        if repo_dir.exists():
            if progress:
                progress(f"[prepare-source] fetching existing cache repo={task.repo} path={repo_dir}")
            self._run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
            return repo_dir
        repo_url = f"https://github.com/{task.repo}.git"
        if progress:
            progress(f"[prepare-source] cloning repo={task.repo} url={repo_url} into cache={repo_dir}")
        self._run(["git", "clone", repo_url, str(repo_dir)], cwd=self._repo_cache_dir)
        return repo_dir

    @staticmethod
    def _clone_workspace(source_repo: Path, workspace_dir: Path) -> None:
        WorkspacePreparer._run(
            ["git", "clone", "--quiet", str(source_repo.resolve()), str(workspace_dir.resolve())],
            cwd=Path.cwd(),
        )

    @staticmethod
    def _run(command: list[str], *, cwd: Path) -> None:
        completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or "<no stderr>"
            stdout = completed.stdout.strip() or "<no stdout>"
            raise RuntimeError(
                f"Command {command!r} failed with exit status {completed.returncode}.\n"
                f"cwd={cwd}\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
            )

    @staticmethod
    def _run_captured(command: list[str], *, cwd: Path) -> str:
        """Run a command and return combined stdout+stderr; raise RuntimeError on non-zero exit."""
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = completed.stdout or ""
        if completed.returncode != 0:
            raise RuntimeError(
                f"Command {command!r} failed (exit {completed.returncode}).\n"
                f"cwd={cwd}\noutput:\n{output.strip() or '<no output>'}"
            )
        return output


class OfficialHarnessGrader:
    """Run grading commands and normalize the results."""

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config

    def grade(self, task: TaskInstance, record: RunRecord, predictions_path: Path, grade_path: Path) -> GradeResult:
        run_dir = Path(record.run_dir).resolve()
        predictions_path = predictions_path.resolve()
        grade_path = grade_path.resolve()
        grade_path.parent.mkdir(parents=True, exist_ok=True)
        if grade_path.exists():
            grade_path.unlink()
        command = self._build_command(task, record, predictions_path, grade_path)
        env = os.environ.copy()
        env.update(self._config.extra_env)
        completed = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        grade_stdout = run_dir / "grade.stdout.log"
        grade_stderr = run_dir / "grade.stderr.log"
        grade_stdout.write_text(completed.stdout, encoding="utf-8")
        grade_stderr.write_text(completed.stderr, encoding="utf-8")
        if grade_path.exists():
            raw = json.loads(grade_path.read_text(encoding="utf-8"))
            resolved = bool(raw.get("resolved"))
            status = raw.get("status", "graded")
            details = raw
        else:
            swebench_report = run_dir / f"{record.arm.value}.{record.run_id}.json"
            if swebench_report.exists():
                raw = json.loads(swebench_report.read_text(encoding="utf-8"))
                resolved = task.instance_id in raw.get("resolved_ids", [])
                status = "graded"
                details = {
                    "returncode": completed.returncode,
                    "stdout_path": str(grade_stdout),
                    "stderr_path": str(grade_stderr),
                    "swebench_report_path": str(swebench_report),
                }
            else:
                resolved = completed.returncode == 0
                status = "graded" if completed.returncode == 0 else "grading_failed"
                details = {
                    "returncode": completed.returncode,
                    "stdout_path": str(grade_stdout),
                    "stderr_path": str(grade_stderr),
                }
        if isinstance(details, dict):
            attach_swebench_grading_analysis(
                details,
                run_dir=run_dir,
                record=record,
                task=task,
                resolved=bool(resolved),
            )
        return GradeResult(
            resolved=resolved,
            status=status,
            details=details,
            graded_at=utc_now(),
        )

    def _build_command(
        self,
        task: TaskInstance,
        record: RunRecord,
        predictions_path: Path,
        grade_path: Path,
    ) -> list[str]:
        predictions_path = predictions_path.resolve()
        grade_path = grade_path.resolve()
        if self._config.grade_command:
            mapping = {
                "dataset_name": self._config.dataset_name,
                "dataset_split": self._config.dataset_split,
                "instance_id": task.instance_id,
                "predictions_path": str(predictions_path),
                "run_id": record.run_id,
                "grade_path": str(grade_path),
                "run_dir": str(Path(record.run_dir).resolve()),
            }
            return [item.format_map(mapping) for item in self._config.grade_command]
        return [
            "python3",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self._config.dataset_name,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            "1",
            "--instance_ids",
            task.instance_id,
            "--run_id",
            record.run_id,
            "--namespace",
            self._config.namespace if self._config.namespace is not None else "none",
            "--instance_image_tag",
            self._config.instance_image_tag,
            "--env_image_tag",
            self._config.env_image_tag,
            "--report_dir",
            str(Path(record.run_dir).resolve()),
        ]


def predictions_entry(task: TaskInstance, record: RunRecord) -> dict[str, str]:
    patch_text = ""
    if record.patch_path:
        patch_text = Path(record.patch_path).read_text(encoding="utf-8")
    return {
        "instance_id": task.instance_id,
        "model_name_or_path": record.arm.value,
        "model_patch": patch_text,
    }


def classify_grading_failure(grade: GradeResult) -> FailureClass:
    if grade.resolved:
        return FailureClass.NONE
    return FailureClass.GRADED_FAIL
