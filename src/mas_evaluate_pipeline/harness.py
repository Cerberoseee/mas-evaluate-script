"""Workspace preparation and grading helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .models import FailureClass, GradeResult, HarnessConfig, PreparedTask, RunRecord, TaskInstance, utc_now


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
        return PreparedTask(
            task=task,
            workspace_dir=str(workspace_dir),
            metadata_path=str(metadata_path),
            source_repo_dir=str(source_repo),
        )

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
            resolved = completed.returncode == 0
            status = "graded" if completed.returncode == 0 else "grading_failed"
            details = {
                "returncode": completed.returncode,
                "stdout_path": str(grade_stdout),
                "stderr_path": str(grade_stderr),
            }
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
