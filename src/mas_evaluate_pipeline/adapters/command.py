"""Command-backed adapter implementation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..models import AdapterConfig, FailureClass, RunArtifacts, RunStatus, TaskInstance, TelemetrySummary
from .base import BenchmarkAdapter

# Path inside every SWE-bench Docker container where the repo lives.
_DOCKER_TESTBED = "/testbed"
# Mount point for the host run_dir inside the Docker container.
_DOCKER_RUN_DIR = "/run_dir"


def _write_test_patch(task: TaskInstance, run_dir: Path) -> Path | None:
    """If the dataset shipped a `test_patch`, persist it to `run_dir/test_patch.diff`.

    Mirrors the official SWE-bench evaluation harness, which always applies the
    dataset's gold `test_patch` before running fail_to_pass / pass_to_pass tests.
    Doing it here means the agent never has to invent or edit those test cases.
    Returns the path on success, or None if the manifest has no test_patch.
    """
    patch_text = task.metadata.get("test_patch")
    if not patch_text or not isinstance(patch_text, str):
        return None
    out = (run_dir / "test_patch.diff").resolve()
    out.write_text(patch_text, encoding="utf-8")
    return out


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
    # Subclasses that keep the orchestrator on the host but route individual
    # bash commands into Docker (via MINI_AGENT_USE_DOCKER) set this to True.
    # When False (default) the entire agent command is wrapped in docker run.
    _bash_via_docker_env: bool = False

    def __init__(self, study_config, adapter_config: AdapterConfig) -> None:
        super().__init__(study_config)
        self.adapter_config = adapter_config

    def run(self, *, task: TaskInstance, prompt: str, run_dir: Path, workspace_dir: Path, repeat_index: int) -> RunArtifacts:
        if not self.adapter_config.command:
            raise RuntimeError(f"{self.arm.value} requires a command configuration")

        run_dir = run_dir.resolve()
        workspace_dir = workspace_dir.resolve()

        use_docker = self.study_config.harness.use_docker
        if use_docker and not self._bash_via_docker_env:
            # Standalone arm (e.g. mini_swe_agent): wrap the entire command in
            # docker run so the agent process itself runs inside the container.
            return self._run_docker(task=task, run_dir=run_dir, repeat_index=repeat_index)
        # MAS arm (or no Docker): orchestrator stays on host; MINI_AGENT_USE_DOCKER
        # tells engineer.py to route bash execution into Docker per-command.
        return self._run_host(
            task=task,
            run_dir=run_dir,
            workspace_dir=workspace_dir,
            repeat_index=repeat_index,
            inject_docker_env=use_docker and self._bash_via_docker_env,
        )

    # ── Host execution ─────────────────────────────────────────────────────────

    def _run_host(
        self,
        *,
        task: TaskInstance,
        run_dir: Path,
        workspace_dir: Path,
        repeat_index: int,
        inject_docker_env: bool = False,
    ) -> RunArtifacts:
        board_dir = (run_dir / "project_board").resolve()
        docs_dir = (run_dir / "knowledge_base").resolve()
        prompt_path = (run_dir / "prompt.txt").resolve()
        result_path = (run_dir / "result.json").resolve()
        telemetry_path = (run_dir / "telemetry.json").resolve()
        patch_path = (run_dir / "patch.diff").resolve()
        task_context_path = (run_dir / "task_context.json").resolve()
        test_patch_path = _write_test_patch(task, run_dir)

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
                "MAS_EVAL_TEST_PATCH_PATH": str(test_patch_path) if test_patch_path else "",
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

        if inject_docker_env:
            # Tell engineer.py to route bash commands through DockerEnvironment
            # instead of LocalEnvironment.  MAS_EVAL_TASK_ID is already set
            # above so engineer.py can derive the SWE-bench image name.
            env["MINI_AGENT_USE_DOCKER"] = "1"

        # If the harness pre-provisioned a venv in workspace/venv/, prepend its
        # bin/ to PATH now so the entire subprocess inherits it.  This is more
        # reliable than relying on the agent runtime's per-command PATH injection
        # (which requires detecting the venv from the filesystem at run time).
        if not inject_docker_env:
            _prepend_workspace_venv_to_env(workspace_dir, env)

        command = [item.format_map(mapping) for item in self.adapter_config.command]
        return self._exec(command, run_dir=run_dir, env=env, cwd=self.adapter_config.cwd or Path.cwd())

    # ── Docker execution ───────────────────────────────────────────────────────

    def _run_docker(
        self,
        *,
        task: TaskInstance,
        run_dir: Path,
        repeat_index: int,
    ) -> RunArtifacts:
        """Run the agent command inside the SWE-bench per-instance Docker container.

        Layout inside the container
        ---------------------------
        /testbed     – repo checked out at base_commit (from the Docker image)
        /run_dir     – bind-mounted from host run_dir; used for artifact exchange
                       (patch.diff, prompt.txt, task_context.json, telemetry.json …)

        All workspace-related env vars are remapped from host paths to their
        Docker equivalents so the agent sees a consistent environment.
        """
        harness = self.study_config.harness
        # Docker doesn't allow __ in image names; SWE-bench uses _1776_ substitution.
        docker_id = task.instance_id.replace("__", "_1776_")
        image = f"{harness.docker_image_prefix}.{docker_id}:{harness.instance_image_tag}"

        # Paths inside the container.
        c_run_dir = _DOCKER_RUN_DIR
        c_workspace = _DOCKER_TESTBED
        c_board = f"{c_run_dir}/project_board"
        c_docs = f"{c_run_dir}/knowledge_base"
        c_prompt = f"{c_run_dir}/prompt.txt"
        c_result = f"{c_run_dir}/result.json"
        c_telemetry = f"{c_run_dir}/telemetry.json"
        c_patch = f"{c_run_dir}/patch.diff"
        c_task_ctx = f"{c_run_dir}/task_context.json"
        host_test_patch = _write_test_patch(task, run_dir)
        c_test_patch = f"{c_run_dir}/test_patch.diff" if host_test_patch else ""

        mapping = {
            "task_id": task.instance_id,
            "instance_id": task.instance_id,
            "workspace_dir": c_workspace,
            "run_dir": c_run_dir,
            "board_dir": c_board,
            "docs_dir": c_docs,
            "repeat_index": repeat_index,
            "base_model": self.study_config.base_model,
            "token_budget": self.study_config.total_token_budget,
            "timeout_seconds": self.study_config.timeout_seconds,
            "repo": task.repo,
        }
        agent_command = [item.format_map(mapping) for item in self.adapter_config.command]

        # Env vars forwarded into the container (no host paths leak in).
        container_env: dict[str, str] = {}
        container_env.update(harness.extra_env)
        container_env.update(self.adapter_config.env)
        container_env.update(
            {
                "MAS_EVAL_ARM": self.arm.value,
                "MAS_EVAL_TASK_ID": task.instance_id,
                "MAS_EVAL_REPO": task.repo,
                "MAS_EVAL_WORKSPACE_DIR": c_workspace,
                "MAS_EVAL_RUN_DIR": c_run_dir,
                "MAS_EVAL_REPEAT_INDEX": str(repeat_index),
                "MAS_EVAL_PROBLEM_STATEMENT": task.problem_statement,
                "MAS_EVAL_PROMPT_PATH": c_prompt,
                "MAS_EVAL_RESULT_PATH": c_result,
                "MAS_EVAL_TELEMETRY_PATH": c_telemetry,
                "MAS_EVAL_PATCH_PATH": c_patch,
                "MAS_EVAL_TEST_PATCH_PATH": c_test_patch,
                "MAS_EVAL_TASK_CONTEXT_PATH": c_task_ctx,
                "MAS_EVAL_BASE_MODEL": self.study_config.base_model,
                "MAS_EVAL_TOKEN_BUDGET": str(self.study_config.total_token_budget),
                "MAS_EVAL_TIMEOUT_SECONDS": str(self.study_config.timeout_seconds),
                "MAS_WORKSPACE_PATH": c_workspace,
                "MAS_BOARD_PATH": c_board,
                "MAS_DOCS_PATH": c_docs,
                "WORKSPACE_GIT_ROOT": c_workspace,
            }
        )

        docker_cmd = ["docker", "run", "--rm", "--platform", "linux/amd64"]

        # Mount the host run_dir so artifacts (patch, prompt, task_context …)
        # can be written by the agent and read back on the host after the run.
        docker_cmd += ["-v", f"{run_dir}:{c_run_dir}"]

        # Forward each env var with -e KEY=VALUE.
        for key, value in container_env.items():
            docker_cmd += ["-e", f"{key}={value}"]

        docker_cmd += ["--workdir", c_workspace]
        docker_cmd += [image]
        docker_cmd += agent_command

        host_env = os.environ.copy()
        return self._exec(docker_cmd, run_dir=run_dir, env=host_env, cwd=Path.cwd())

    # ── Shared execution helper ────────────────────────────────────────────────

    def _exec(
        self,
        command: list[str],
        *,
        run_dir: Path,
        env: dict[str, str],
        cwd: Path | str,
    ) -> RunArtifacts:
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
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
