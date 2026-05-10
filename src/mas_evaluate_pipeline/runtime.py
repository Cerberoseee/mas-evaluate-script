"""Benchmark runtime orchestration."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Callable

from .adapters.mas import MasCentralizeAdapter, MasDecentralizedAdapter
from .adapters.mini_swe_agent import MiniSweAgentAdapter
from .adapters.base import prompt_sha256
from .constants import DEFAULT_RUNS_DIR
from .harness import WorkspacePreparer
from .models import (
    ArmName,
    FailureClass,
    RunRecord,
    RunStatus,
    StudyConfig,
    TaskInstance,
    TaskManifest,
    TelemetrySummary,
    utc_now,
)


class BenchmarkRuntime:
    def __init__(self, study_config: StudyConfig) -> None:
        self.study_config = study_config
        self.workspace_preparer = WorkspacePreparer(study_config.harness)

    def run_manifest(
        self,
        manifest: TaskManifest,
        *,
        output_root: Path | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> list[RunRecord]:
        root = output_root or Path(self.study_config.output_root)
        run_id = self.study_config.run_id or utc_now().strftime("%Y%m%dT%H%M%SZ")
        runs_root = root / DEFAULT_RUNS_DIR / run_id
        records: list[RunRecord] = []
        total_runs = len(self.study_config.arms) * len(manifest.tasks) * self.study_config.repeats
        completed_runs = 0
        if progress:
            progress(
                f"[run-benchmark] run_id={run_id} arms={len(self.study_config.arms)} "
                f"tasks={len(manifest.tasks)} repeats={self.study_config.repeats} total_runs={total_runs}"
            )
        for arm in self.study_config.arms:
            adapter = self._make_adapter(arm)
            for task in manifest.tasks:
                for repeat_index in range(self.study_config.repeats):
                    completed_runs += 1
                    run_label = (
                        f"{completed_runs}/{total_runs} arm={arm.value} "
                        f"task={task.instance_id} repeat={repeat_index + 1}"
                    )
                    run_dir = runs_root / arm.value / task.instance_id / f"repeat_{repeat_index + 1}"
                    record_path = run_dir / "run_record.json"
                    if record_path.exists():
                        if progress:
                            progress(f"[skip] {run_label} existing_record={record_path}")
                        records.append(RunRecord.model_validate_json(record_path.read_text(encoding="utf-8")))
                        continue
                    workspace_dir = run_dir / "workspace"
                    prompt = build_prompt(
                        task, manifest.dataset_name, manifest.dataset_split, workspace_dir=workspace_dir
                    )
                    prompt_digest = prompt_sha256(prompt)
                    if progress:
                        progress(f"[start] {run_label}")
                    run_dir.mkdir(parents=True, exist_ok=True)
                    prompt_path = run_dir / "prompt.txt"
                    prompt_path.write_text(prompt, encoding="utf-8")
                    task_context_path = run_dir / "task_context.json"
                    task_context_path.write_text(
                        json.dumps(task.model_dump(), indent=2),
                        encoding="utf-8",
                    )
                    docs_metadata_path = run_dir / "task_metadata.json"
                    started_at = utc_now()
                    if progress:
                        progress(f"[prepare] {run_label} workspace={workspace_dir}")
                    try:
                        self.workspace_preparer.prepare(
                            task,
                            workspace_dir,
                            docs_metadata_path,
                            progress=progress,
                        )
                    except Exception as exc:  # noqa: BLE001
                        record = RunRecord(
                            run_id=run_id,
                            arm=arm,
                            instance_id=task.instance_id,
                            repeat_index=repeat_index,
                            status=RunStatus.FAILED,
                            failure_class=FailureClass.REPOSITORY_PREPARATION,
                            note=str(exc),
                            started_at=started_at,
                            completed_at=utc_now(),
                            duration_seconds=0.0,
                            workspace_dir=str(workspace_dir),
                            run_dir=str(run_dir),
                            prompt_path=str(prompt_path),
                            prompt_sha256=prompt_digest,
                            telemetry=TelemetrySummary(),
                        )
                        record_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
                        records.append(record)
                        if progress:
                            progress(f"[failed] {run_label} failure_class={record.failure_class.value} note={record.note}")
                        continue
                    if progress:
                        progress(f"[run] {run_label} run_dir={run_dir}")
                    artifacts = adapter.run(
                        task=task,
                        prompt=prompt,
                        run_dir=run_dir,
                        workspace_dir=workspace_dir,
                        repeat_index=repeat_index,
                    )
                    completed_at = utc_now()
                    record = RunRecord(
                        run_id=run_id,
                        arm=arm,
                        instance_id=task.instance_id,
                        repeat_index=repeat_index,
                        status=artifacts.status,
                        failure_class=artifacts.failure_class,
                        note=artifacts.note,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_seconds=(completed_at - started_at) / timedelta(seconds=1),
                        workspace_dir=str(workspace_dir),
                        run_dir=str(run_dir),
                        prompt_path=str(prompt_path),
                        patch_path=artifacts.patch_path,
                        stdout_path=artifacts.stdout_path,
                        stderr_path=artifacts.stderr_path,
                        telemetry_path=artifacts.telemetry_path,
                        result_path=artifacts.result_path,
                        prompt_sha256=prompt_digest,
                        telemetry=artifacts.telemetry,
                    )
                    record_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
                    records.append(record)
                    if progress:
                        progress(
                            f"[done] {run_label} status={record.status.value} "
                            f"failure_class={record.failure_class.value} "
                            f"duration_s={record.duration_seconds:.2f} "
                            f"tokens={record.telemetry.total_tokens}"
                        )
        return records

    def _make_adapter(self, arm: ArmName):
        adapter_config = self.study_config.adapter_config_for(arm)
        if arm is ArmName.MINI_SWE_AGENT:
            return MiniSweAgentAdapter(self.study_config, adapter_config)
        if arm is ArmName.MAS_CENTRALIZE:
            return MasCentralizeAdapter(self.study_config, adapter_config)
        if arm is ArmName.MAS_DECENTRALIZED:
            return MasDecentralizedAdapter(self.study_config, adapter_config)
        raise ValueError(f"Unsupported arm: {arm}")


def build_prompt(
    task: TaskInstance,
    dataset_name: str,
    dataset_split: str,
    *,
    workspace_dir: Path | None = None,
) -> str:
    hints = task.hints_text or "None"
    fail_to_pass = "\n".join(f"- {test}" for test in task.fail_to_pass) or "- Not provided"
    pass_to_pass = "\n".join(f"- {test}" for test in task.pass_to_pass) or "- Not provided"
    workspace_line = f"Workspace path: {workspace_dir}\n" if workspace_dir is not None else ""
    return (
        f"SWE-bench dataset: {dataset_name} ({dataset_split})\n"
        f"Instance ID: {task.instance_id}\n"
        f"Repository: {task.repo}\n"
        f"Base commit: {task.base_commit or 'unknown'}\n"
        f"Version: {task.version or 'unknown'}\n"
        f"{workspace_line}"
        f"\nProblem statement:\n{task.problem_statement}\n\n"
        f"Hints:\n{hints}\n\n"
        f"Fail-to-pass tests:\n{fail_to_pass}\n\n"
        f"Pass-to-pass tests:\n{pass_to_pass}\n"
    )
