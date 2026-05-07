from __future__ import annotations

from pathlib import Path

from mas_evaluate_pipeline.harness import OfficialHarnessGrader
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
