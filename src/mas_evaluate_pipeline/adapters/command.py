"""Command-backed adapter implementation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..models import AdapterConfig, FailureClass, RunArtifacts, RunStatus, TaskInstance, TelemetrySummary
from .base import BenchmarkAdapter


def _prepend_workspace_venv_to_env(workspace_dir: Path, env: dict[str, str]) -> None:
    """If workspace_dir/venv/bin exists, prepend it to PATH in *env*.

    This injects the pre-provisioned venv into the subprocess's process
    environment so every tool the agent spawns (pytest, pip, …) resolves
    from the venv without relying on in-process PATH patching inside the
    agent runtime.
    """
    for venv_name in ("venv", ".venv"):
        bin_dir = workspace_dir / venv_name / "bin"
        if bin_dir.is_dir():
            venv_str = str(bin_dir)
            current_path = env.get("PATH", "")
            if not (current_path == venv_str or current_path.startswith(venv_str + os.pathsep)):
                env["PATH"] = f"{venv_str}{os.pathsep}{current_path}"
            env.setdefault("VIRTUAL_ENV", str(bin_dir.parent))
            return


class CommandAdapter(BenchmarkAdapter):
    def __init__(self, study_config, adapter_config: AdapterConfig) -> None:
        super().__init__(study_config)
        self.adapter_config = adapter_config

    def run(self, *, task: TaskInstance, prompt: str, run_dir: Path, workspace_dir: Path, repeat_index: int) -> RunArtifacts:
        if not self.adapter_config.command:
            raise RuntimeError(f"{self.arm.value} requires a command configuration")

        run_dir = run_dir.resolve()
        workspace_dir = workspace_dir.resolve()
        board_dir = (run_dir / "project_board").resolve()
        docs_dir = (run_dir / "knowledge_base").resolve()
        prompt_path = (run_dir / "prompt.txt").resolve()
        result_path = (run_dir / "result.json").resolve()
        telemetry_path = (run_dir / "telemetry.json").resolve()
        patch_path = (run_dir / "patch.diff").resolve()
        task_context_path = (run_dir / "task_context.json").resolve()
        env = os.environ.copy()
        env.update(self.study_config.harness.extra_env)
        env.update(self.adapter_config.env)
        mapping = {
            "task_id": task.instance_id,
            "instance_id": task.instance_id,
            "workspace_dir": str(workspace_dir),
            "run_dir": str(run_dir),
            "board_dir": str(board_dir),
            "docs_dir": str(docs_dir),
            "repeat_index": repeat_index,
            "base_model": self.study_config.base_model,
            "token_budget": self.study_config.total_token_budget,
            "timeout_seconds": self.study_config.timeout_seconds,
            "repo": task.repo,
        }
        env.update(
            {
                "MAS_EVAL_ARM": self.arm.value,
                "MAS_EVAL_TASK_ID": task.instance_id,
                "MAS_EVAL_REPO": task.repo,
                "MAS_EVAL_WORKSPACE_DIR": str(workspace_dir),
                "MAS_EVAL_RUN_DIR": str(run_dir),
                "MAS_EVAL_REPEAT_INDEX": str(repeat_index),
                "MAS_EVAL_PROBLEM_STATEMENT": task.problem_statement,
                "MAS_EVAL_PROMPT_PATH": str(prompt_path),
                "MAS_EVAL_RESULT_PATH": str(result_path),
                "MAS_EVAL_TELEMETRY_PATH": str(telemetry_path),
                "MAS_EVAL_PATCH_PATH": str(patch_path),
                "MAS_EVAL_TASK_CONTEXT_PATH": str(task_context_path),
                "MAS_EVAL_BASE_MODEL": self.study_config.base_model,
                "MAS_EVAL_TOKEN_BUDGET": str(self.study_config.total_token_budget),
                "MAS_EVAL_TIMEOUT_SECONDS": str(self.study_config.timeout_seconds),
                "MAS_WORKSPACE_PATH": str(workspace_dir),
                "MAS_BOARD_PATH": str(board_dir),
                "MAS_DOCS_PATH": str(docs_dir),
                "WORKSPACE_GIT_ROOT": str(workspace_dir),
            }
        )

        # If the harness pre-provisioned a venv in workspace/venv/, prepend its
        # bin/ to PATH now so the entire subprocess inherits it.  This is more
        # reliable than relying on the agent runtime's per-command PATH injection
        # (which requires detecting the venv from the filesystem at run time).
        _prepend_workspace_venv_to_env(workspace_dir, env)

        command = [item.format_map(mapping) for item in self.adapter_config.command]
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        try:
            completed = subprocess.run(
                command,
                cwd=self.adapter_config.cwd or Path.cwd(),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.study_config.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("Command timed out.", encoding="utf-8")
            return RunArtifacts(
                status=RunStatus.FAILED,
                failure_class=FailureClass.TIMEOUT,
                note="Adapter command timed out.",
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        telemetry_path = run_dir / "telemetry.json"
        result_path = run_dir / "result.json"
        patch_path = run_dir / "patch.diff"
        telemetry = _load_telemetry(telemetry_path)
        if completed.returncode != 0:
            return RunArtifacts(
                status=RunStatus.FAILED,
                failure_class=FailureClass.ADAPTER_RUNTIME,
                note=f"Command exited with status {completed.returncode}.",
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                telemetry_path=str(telemetry_path) if telemetry_path.exists() else None,
                result_path=str(result_path) if result_path.exists() else None,
                telemetry=telemetry,
            )
        if not patch_path.exists():
            return RunArtifacts(
                status=RunStatus.FAILED,
                failure_class=FailureClass.NO_PATCH,
                note="Adapter completed without producing patch.diff.",
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                telemetry_path=str(telemetry_path) if telemetry_path.exists() else None,
                result_path=str(result_path) if result_path.exists() else None,
                telemetry=telemetry,
            )
        return RunArtifacts(
            status=RunStatus.SUCCESS,
            patch_path=str(patch_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            telemetry_path=str(telemetry_path) if telemetry_path.exists() else None,
            result_path=str(result_path) if result_path.exists() else None,
            telemetry=telemetry,
        )


def _load_telemetry(path: Path) -> TelemetrySummary:
    if not path.exists():
        return TelemetrySummary()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TelemetrySummary.model_validate(raw)
