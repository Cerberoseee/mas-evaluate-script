from __future__ import annotations

import json
from pathlib import Path

from mas_evaluate_pipeline.harness import OfficialHarnessGrader, summarize_swebench_instance_tests
from mas_evaluate_pipeline.models import ArmName, GradeResult, HarnessConfig, RunRecord, RunStatus


def test_grade_command_uses_absolute_paths_for_relative_run_dir(tmp_path: Path) -> None:
    grader = OfficialHarnessGrader(
        HarnessConfig(
            grade_command=[
                "python3",
                "-m",
                "fake.module",
                "--predictions_path",
                "{predictions_path}",
                "--grade_path",
                "{grade_path}",
                "--run_dir",
                "{run_dir}",
            ]
        )
    )
    relative_run_dir = Path("runs") / "r1" / "mas_decentralized" / "task" / "repeat_1"
    record = RunRecord(
        run_id="r1",
        arm=ArmName.MAS_DECENTRALIZED,
        instance_id="repo__issue-1",
        repeat_index=0,
        status=RunStatus.SUCCESS,
        workspace_dir=str(relative_run_dir / "workspace"),
        run_dir=str(relative_run_dir),
        prompt_path=str(relative_run_dir / "prompt.txt"),
        grade=GradeResult(),
    )
    command = grader._build_command(
        task=type("Task", (), {"instance_id": "repo__issue-1"})(),
        record=record,
        predictions_path=relative_run_dir / "predictions.jsonl",
        grade_path=relative_run_dir / "grade-result.json",
    )
    pred_index = command.index("--predictions_path") + 1
    grade_index = command.index("--grade_path") + 1
    run_dir_index = command.index("--run_dir") + 1
    assert Path(command[pred_index]).is_absolute()
    assert Path(command[grade_index]).is_absolute()
    assert Path(command[run_dir_index]).is_absolute()


def test_default_grade_command_uses_local_namespace(tmp_path: Path) -> None:
    grader = OfficialHarnessGrader(HarnessConfig())
    run_dir = Path("runs") / "r1" / "mas_centralize" / "task" / "repeat_1"
    record = RunRecord(
        run_id="r1",
        arm=ArmName.MAS_CENTRALIZE,
        instance_id="repo__issue-1",
        repeat_index=0,
        status=RunStatus.SUCCESS,
        workspace_dir=str(run_dir / "workspace"),
        run_dir=str(run_dir),
        prompt_path=str(run_dir / "prompt.txt"),
        grade=GradeResult(),
    )
    task = type("Task", (), {"instance_id": "repo__issue-1"})()
    command = grader._build_command(
        task=task,
        record=record,
        predictions_path=run_dir / "predictions.jsonl",
        grade_path=run_dir / "grade-result.json",
    )
    assert "--namespace" in command
    assert command[command.index("--namespace") + 1] == "none"
    assert "--report_dir" in command


def test_summarize_swebench_instance_tests_reports_fail_buckets() -> None:
    payload = {
        "org__proj-42": {
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": ["passed_bugfix.py::test_a"],
                    "failure": ["model/tests/test_bug.py::test_still_bad"],
                },
                "PASS_TO_PASS": {"success": ["ok.py::t1"], "failure": ["ok.py::regressed"]},
                "PASS_TO_FAIL": {"success": [], "failure": []},
                "FAIL_TO_FAIL": {"success": [], "failure": []},
            }
        }
    }
    summary = summarize_swebench_instance_tests(payload, "org__proj-42")
    assert summary["fail_to_pass_still_failing"] == ["model/tests/test_bug.py::test_still_bad"]
    assert summary["pass_to_pass_regressions"] == ["ok.py::regressed"]
    assert "fail_to_pass test(s) still failing" in summary["summary"]
    assert "pass_to_pass test(s) regressed" in summary["summary"]
    assert "FAIL_TO_PASS" in summary["tests_status"]


def test_find_swebench_instance_report_prefers_canonical_path(tmp_path: Path) -> None:
    from mas_evaluate_pipeline.harness import find_swebench_instance_report

    instance_id = "repo__issue-1"
    report_path = (
        tmp_path
        / "logs"
        / "run_evaluation"
        / "study-1"
        / "mas_centralize"
        / instance_id
        / "report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({instance_id: {"tests_status": {}}}), encoding="utf-8")

    record = RunRecord(
        run_id="study-1",
        arm=ArmName.MAS_CENTRALIZE,
        instance_id=instance_id,
        repeat_index=0,
        status=RunStatus.SUCCESS,
        workspace_dir=str(tmp_path / "workspace"),
        run_dir=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.txt"),
        grade=GradeResult(),
    )
    found = find_swebench_instance_report(tmp_path, record, instance_id)
    assert found == report_path
