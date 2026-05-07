from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from mas_evaluate_pipeline.cli import app, load_study_config
from mas_evaluate_pipeline.reporting import load_run_records


def init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "foo.py").write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True, text=True)


def test_cli_prepare_run_grade_report(tmp_path: Path) -> None:
    repo_dir = tmp_path / "mirror"
    init_repo(repo_dir)
    config_path = tmp_path / "study.toml"
    config_path.write_text(
        "\n".join(
            [
                'run_id = "cli-run"',
                f'output_root = "{tmp_path}"',
                "repeats = 1",
                'arms = ["mas_centralize"]',
                "",
                "[harness]",
                f'repo_mirrors = {{ "example/repo" = "{repo_dir}" }}',
                f'grade_command = ["{sys.executable}", "tests/fixtures/fake_grader.py", "{{predictions_path}}", "{{grade_path}}"]',
                "",
                "[adapters.mas_centralize]",
                f'command = ["{sys.executable}", "tests/fixtures/fake_adapter_runner.py"]',
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    result = runner.invoke(
        app,
        [
            "prepare-suite",
            "--manifest-path",
            str(manifest_path),
            "--dataset-jsonl",
            "tests/data/sample_tasks.jsonl",
            "--count",
            "1",
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(app, ["run-benchmark", "--config", str(config_path), "--manifest-path", str(manifest_path)])
    assert result.exit_code == 0
    result = runner.invoke(
        app,
        ["grade-runs", "--config", str(config_path), "--manifest-path", str(manifest_path), "--run-id", "cli-run"],
    )
    assert result.exit_code == 0
    result = runner.invoke(app, ["report", "--config", str(config_path), "--run-id", "cli-run"])
    assert result.exit_code == 0
    assert (tmp_path / "reports" / "cli-run" / "report.md").exists()


def test_load_study_config_reads_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    config_path = tmp_path / "study.toml"
    config_path.write_text(
        "\n".join(
            [
                'run_id = "env-run"',
                'env_file = ".env"',
            ]
        ),
        encoding="utf-8",
    )
    os.environ.pop("OPENAI_API_KEY", None)
    config = load_study_config(config_path)
    assert config.env_file == ".env"
    assert os.environ["OPENAI_API_KEY"] == "test-key"


def test_grade_runs_retries_after_grading_failure(tmp_path: Path) -> None:
    repo_dir = tmp_path / "mirror"
    init_repo(repo_dir)
    config_path = tmp_path / "study.toml"
    failing_grader = "import sys; sys.stderr.write('boom\\n'); raise SystemExit(1)"
    succeeding_grader = (
        "import json, sys; "
        "from pathlib import Path; "
        "Path(sys.argv[2]).write_text(json.dumps({{'resolved': True, 'status': 'graded'}}), encoding='utf-8')"
    )
    config_path.write_text(
        "\n".join(
            [
                'run_id = "cli-run"',
                f'output_root = "{tmp_path}"',
                "repeats = 1",
                'arms = ["mas_centralize"]',
                "",
                "[harness]",
                f'repo_mirrors = {{ "example/repo" = "{repo_dir}" }}',
                f'grade_command = ["{sys.executable}", "-c", "{failing_grader}", "{{predictions_path}}", "{{grade_path}}"]',
                "",
                "[adapters.mas_centralize]",
                f'command = ["{sys.executable}", "tests/fixtures/fake_adapter_runner.py"]',
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    result = runner.invoke(
        app,
        [
            "prepare-suite",
            "--manifest-path",
            str(manifest_path),
            "--dataset-jsonl",
            "tests/data/sample_tasks.jsonl",
            "--count",
            "1",
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(app, ["run-benchmark", "--config", str(config_path), "--manifest-path", str(manifest_path)])
    assert result.exit_code == 0
    result = runner.invoke(
        app,
        ["grade-runs", "--config", str(config_path), "--manifest-path", str(manifest_path), "--run-id", "cli-run"],
    )
    assert result.exit_code == 0
    records = load_run_records(tmp_path / "runs" / "cli-run")
    assert records[0].grade.status == "grading_failed"

    config_path.write_text(
        "\n".join(
            [
                'run_id = "cli-run"',
                f'output_root = "{tmp_path}"',
                "repeats = 1",
                'arms = ["mas_centralize"]',
                "",
                "[harness]",
                f'repo_mirrors = {{ "example/repo" = "{repo_dir}" }}',
                f'grade_command = ["{sys.executable}", "-c", "{succeeding_grader}", "{{predictions_path}}", "{{grade_path}}"]',
                "",
                "[adapters.mas_centralize]",
                f'command = ["{sys.executable}", "tests/fixtures/fake_adapter_runner.py"]',
            ]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["grade-runs", "--config", str(config_path), "--manifest-path", str(manifest_path), "--run-id", "cli-run"],
    )
    assert result.exit_code == 0
    records = load_run_records(tmp_path / "runs" / "cli-run")
    assert records[0].grade.status == "graded"
    assert records[0].grade.resolved is True
