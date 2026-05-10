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
# Strategy (mirrors the official SWE-bench harness Docker build):
#
#   1. Create a uv-managed venv with the correct Python version.
#   2. Pre-install universal build tools so compiled packages can build.
#   3. If the task has an environment_setup_commit:
#        a. git checkout that commit inside the workspace clone.
#        b. pip install -e .[test]  (or fall back to -e .)  with --no-build-isolation
#           so the pinned build tools above are used by the build backend.
#        c. git checkout back to base_commit.
#      The .so files compiled from environment_setup_commit stay in the venv
#      exactly as they would in the official Docker image.
#   4. If no environment_setup_commit, attempt the same editable install from
#      whatever commit is currently checked out (best-effort).

# Build tools pre-installed before any editable install.
#
# setuptools<67.3: dep_util (used by several setup_package.py files of that era)
#   was removed in 67.3.0.  Pinned here so --no-build-isolation picks it up
#   instead of letting pip/uv resolve the latest in an isolated build env.
# numpy<2: numpy 2.x tightened PyArray_* argument types; C extensions written
#   against numpy 1.x headers fail to compile with numpy 2.x headers.
# hypothesis: omitted from many [test] extras but required by several test suites.
_BUILD_PRE_DEPS: list[str] = [
    "cython",
    "wheel",
    "setuptools<67.3",
    "numpy<2",
    "extension-helpers",
    "hypothesis",
    "pytest",
    "packaging",
]

# Environment variables injected for every editable-install subprocess.
# GCC ≥ 14 (Ubuntu 24.04) promotes -Wincompatible-pointer-types to a hard error
# in gnu99 mode.  The SWE-bench Docker was built on Ubuntu 22.04 / GCC 11 where
# it was only a warning.  Suppress it unconditionally — harmless on older GCC.
_BUILD_ENV: dict[str, str] = {
    "CFLAGS": "-Wno-incompatible-pointer-types",
}


def _swebench_docker_image(instance_id: str, *, prefix: str, tag: str) -> str:
    """Return the Docker image name for a SWE-bench instance.

    The SWE-bench project replaces ``__`` with ``_1776_`` when publishing
    images because Docker doesn't allow consecutive underscores in names.
    """
    docker_id = instance_id.replace("__", "_1776_")
    return f"{prefix}.{docker_id}:{tag}"


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
        if self._config.use_docker:
            return self._prepare_docker(task, workspace_dir, metadata_path, progress=progress)
        return self._prepare_host(task, workspace_dir, metadata_path, progress=progress)

    def _prepare_host(
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
        self._write_task_metadata(metadata_path, task)
        if self._config.auto_provision_runtime:
            self._provision_runtime(workspace_dir, task, progress=progress)
        return PreparedTask(
            task=task,
            workspace_dir=str(workspace_dir),
            metadata_path=str(metadata_path),
            source_repo_dir=str(source_repo),
        )

    def _prepare_docker(
        self,
        task: TaskInstance,
        workspace_dir: Path,
        metadata_path: Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> PreparedTask:
        """Pull the official SWE-bench per-instance Docker image.

        No host-side git clone or venv provisioning is done — the container
        already has the repo checked out at base_commit with every C extension
        compiled against the original Ubuntu 22.04 / GCC 11 toolchain.

        The host-side workspace_dir is created as an empty marker so the rest
        of the pipeline has a stable path reference.
        """
        image = _swebench_docker_image(
            task.instance_id,
            prefix=self._config.docker_image_prefix,
            tag=self._config.instance_image_tag,
        )
        if progress:
            progress(f"[prepare-docker] pulling image={image}")
        self._run(
            ["docker", "pull", "--platform", "linux/amd64", image],
            cwd=Path.cwd(),
        )
        if progress:
            progress(f"[prepare-docker] image ready: {image}")
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self._write_task_metadata(metadata_path, task)
        return PreparedTask(
            task=task,
            workspace_dir=str(workspace_dir),
            metadata_path=str(metadata_path),
            source_repo_dir=None,
        )

    @staticmethod
    def _write_task_metadata(metadata_path: Path, task: TaskInstance) -> None:
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

    def _provision_runtime(
        self,
        workspace_dir: Path,
        task: TaskInstance,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Create workspace/venv/ and install dependencies via editable install.

        Mirrors the official SWE-bench Docker build:
          1. Create venv with the configured Python version.
          2. Pre-install universal build tools (_BUILD_PRE_DEPS).
          3. If task.environment_setup_commit is set, check out that commit,
             run `pip install -e .[test]` (falling back to `-e .`), then restore
             base_commit.  The compiled .so files stay in the venv just as they
             would in the official Docker image.
          4. If no environment_setup_commit, run the same editable install from
             the current commit (best-effort).

        All steps after venv creation are best-effort: failures are logged but
        do not abort the run.
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
                extra_env=_BUILD_ENV,
            )

        def _git(*args: str) -> None:
            self._run_captured(["git"] + list(args), cwd=workspace_dir)

        # ── Step 1: resolve Python and create venv ─────────────────────────────
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

        # ── Step 2: pre-install build tools ────────────────────────────────────
        pre_deps = list(_BUILD_PRE_DEPS) + list(self._config.provision_extra_packages)
        try:
            _log(f"installing build tools: {pre_deps}")
            _uv_pip(*pre_deps)
        except Exception as exc:  # noqa: BLE001
            _log(f"WARNING: build tool pre-install failed — {exc}")

        # ── Step 3: editable install from environment_setup_commit ─────────────
        # Check out the commit used when the official Docker image was built,
        # install the package (compiling all C/Cython extensions against the
        # venv's pinned build tools), then restore base_commit.  The .so files
        # compiled here persist in the venv regardless of which source commit is
        # active, exactly matching the Docker environment.
        env_commit = task.environment_setup_commit
        base_commit = task.base_commit

        if env_commit:
            _log(f"checking out environment_setup_commit {env_commit}")
            try:
                _git("checkout", env_commit)
            except Exception as exc:  # noqa: BLE001
                _log(f"WARNING: could not checkout environment_setup_commit — {exc}; installing from current commit")
                env_commit = None  # fall through to best-effort install below

        install_ok = False
        for install_spec in [".[test]", "."]:
            try:
                _log(f"running editable install: pip install -e {install_spec} --no-build-isolation")
                _uv_pip("-e", install_spec, "--no-build-isolation", "--verbose")
                _log(f"editable install complete ({install_spec})")
                install_ok = True
                break
            except Exception as exc:  # noqa: BLE001
                if install_spec == ".[test]":
                    _log(f".[test] extras failed ({exc}), retrying with bare -e .")
                else:
                    _log(f"WARNING: editable install failed — {exc}; agent may lack compiled extensions")

        if env_commit and base_commit:
            try:
                _log(f"restoring base_commit {base_commit}")
                _git("checkout", base_commit)
            except Exception as exc:  # noqa: BLE001
                _log(f"WARNING: could not restore base_commit — {exc}")

        _ = install_ok  # reported via logs above

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
    def _run_captured(
        command: list[str],
        *,
        cwd: Path,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        """Run a command and return combined stdout+stderr; raise RuntimeError on non-zero exit."""
        env: dict[str, str] | None = None
        if extra_env:
            env = {**os.environ, **extra_env}
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
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
