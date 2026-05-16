"""Typer CLI for the evaluation pipeline."""

from __future__ import annotations

import json
import logging
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


def _grade_runs_logger(runs_root: Path) -> tuple[logging.Logger, list[logging.Handler]]:
    runs_root.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"{__name__}.grade_runs")
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(runs_root / "grade_runs.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    return log, [file_handler, stream_handler]


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
    log, handlers = _grade_runs_logger(runs_root)
    try:
        log.info(
            "grade-runs started run_id=%s runs_root=%s record_count=%d",
            run_id,
            runs_root,
            len(records),
        )
        grader = OfficialHarnessGrader(study_config.harness)
        for record in records:
            print(record.model_dump_json(indent=2))
            task = task_lookup[record.instance_id]
            if record.status != record.status.SUCCESS:
                if record.grade.status == "skipped":
                    log.info(
                        "skip instance_id=%s reason=run_not_success grade_already_skipped",
                        record.instance_id,
                    )
                    continue
                log.info(
                    "mark skipped instance_id=%s run_status=%s",
                    record.instance_id,
                    record.status.value if hasattr(record.status, "value") else record.status,
                )
                record.grade.status = "skipped"
                record.grade.resolved = False
                record.failure_class = record.failure_class or FailureClass.GRADED_FAIL
            else:
                if record.grade.status == "graded":
                    log.info(
                        "skip instance_id=%s reason=already_graded resolved=%s",
                        record.instance_id,
                        record.grade.resolved,
                    )
                    continue
                log.info("grading instance_id=%s run_dir=%s", record.instance_id, record.run_dir)
                predictions_path = (Path(record.run_dir) / "predictions.jsonl").resolve()
                predictions_path.write_text(json.dumps(predictions_entry(task, record)) + "\n", encoding="utf-8")
                grade_path = (Path(record.run_dir) / "grade-result.json").resolve()
                record.grade = grader.grade(task, record, predictions_path, grade_path)
                record.grade_path = str(grade_path)
                log.info(
                    "graded instance_id=%s status=%s resolved=%s",
                    record.instance_id,
                    record.grade.status,
                    record.grade.resolved,
                )
                if not record.grade.resolved:
                    record.failure_class = classify_grading_failure(record.grade)
            Path(record.run_dir, "run_record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        log.info("grade-runs finished run_id=%s", run_id)
    finally:
        for h in handlers:
            log.removeHandler(h)
            h.close()
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
