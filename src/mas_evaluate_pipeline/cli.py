"""Typer CLI for the evaluation pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from .constants import DEFAULT_MANIFEST_PATH, DEFAULT_TASK_COUNT
from .harness import OfficialHarnessGrader, classify_grading_failure, predictions_entry
from .manifest import build_task_manifest, load_dataset_records, load_dataset_records_from_jsonl, load_manifest, save_manifest
from .models import FailureClass, StudyConfig
from .reporting import load_run_records, write_report
from .runtime import BenchmarkRuntime


app = typer.Typer(help="SWE-bench communication benchmark pipeline.")


@app.command("prepare-suite")
def prepare_suite(
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST_PATH, "--manifest-path"),
    dataset_name: str = typer.Option("SWE-bench/SWE-bench_Lite", "--dataset-name"),
    dataset_split: str = typer.Option("test", "--dataset-split"),
    dataset_jsonl: str = typer.Option("", "--dataset-jsonl"),
    count: int = typer.Option(DEFAULT_TASK_COUNT, "--count"),
) -> None:
    records = (
        load_dataset_records_from_jsonl(Path(dataset_jsonl))
        if dataset_jsonl
        else load_dataset_records(dataset_name, dataset_split)
    )
    manifest = build_task_manifest(
        records,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        count=count,
    )
    save_manifest(manifest, manifest_path)
    typer.echo(f"Wrote manifest with {len(manifest.tasks)} tasks to {manifest_path}")


@app.command("run-benchmark")
def run_benchmark(
    config_path: Path = typer.Option(..., "--config"),
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST_PATH, "--manifest-path"),
) -> None:
    study_config = load_study_config(config_path)
    manifest = load_manifest(manifest_path)
    runtime = BenchmarkRuntime(study_config)
    typer.echo(f"Loaded config from {config_path}")
    typer.echo(f"Loaded manifest from {manifest_path} with {len(manifest.tasks)} tasks")
    records = runtime.run_manifest(manifest, progress=typer.echo)
    typer.echo(f"Completed or reused {len(records)} run records.")


@app.command("grade-runs")
def grade_runs(
    config_path: Path = typer.Option(..., "--config"),
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST_PATH, "--manifest-path"),
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    study_config = load_study_config(config_path)
    manifest = load_manifest(manifest_path)
    task_lookup = {task.instance_id: task for task in manifest.tasks}
    runs_root = Path(study_config.output_root) / "runs" / run_id
    records = load_run_records(runs_root)
    grader = OfficialHarnessGrader(study_config.harness)
    for record in records:
        task = task_lookup[record.instance_id]
        if record.status != record.status.SUCCESS:
            if record.grade.status == "skipped":
                continue
            record.grade.status = "skipped"
            record.grade.resolved = False
            record.failure_class = record.failure_class or FailureClass.GRADED_FAIL
        else:
            if record.grade.status == "graded":
                continue
            predictions_path = (Path(record.run_dir) / "predictions.jsonl").resolve()
            predictions_path.write_text(json.dumps(predictions_entry(task, record)) + "\n", encoding="utf-8")
            grade_path = (Path(record.run_dir) / "grade-result.json").resolve()
            record.grade = grader.grade(task, record, predictions_path, grade_path)
            record.grade_path = str(grade_path)
            if not record.grade.resolved:
                record.failure_class = classify_grading_failure(record.grade)
        Path(record.run_dir, "run_record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Graded runs under {runs_root}")


@app.command("report")
def report(
    config_path: Path = typer.Option(..., "--config"),
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    study_config = load_study_config(config_path)
    runs_root = Path(study_config.output_root) / "runs" / run_id
    records = load_run_records(runs_root)
    report_dir = write_report(records, output_root=Path(study_config.output_root), run_id=run_id)
    typer.echo(f"Wrote report to {report_dir}")


def load_study_config(path: Path) -> StudyConfig:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(raw)
    else:
        import tomllib

        payload = tomllib.loads(raw)
    config = StudyConfig.model_validate(payload)
    _load_env_file_if_present(config, base_dir=path.parent)
    if config.openai_api_key:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    return config


def _load_env_file_if_present(config: StudyConfig, *, base_dir: Path) -> None:
    if not config.env_file:
        return
    env_path = Path(config.env_file)
    if not env_path.is_absolute():
        env_path = (base_dir / env_path).resolve()
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))
