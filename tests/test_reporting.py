from __future__ import annotations

from pathlib import Path

from mas_evaluate_pipeline.models import ArmName, GradeResult, RunRecord, RunStatus, TelemetrySummary
from mas_evaluate_pipeline.reporting import pairwise_deltas, summarize, write_report


def make_record(tmp_path: Path, arm: ArmName, resolved: bool, repeat_index: int) -> RunRecord:
    return RunRecord(
        run_id="run-1",
        arm=arm,
        instance_id="repo__issue-1",
        repeat_index=repeat_index,
        status=RunStatus.SUCCESS,
        workspace_dir=str(tmp_path / "workspace"),
        run_dir=str(tmp_path / arm.value),
        prompt_path=str(tmp_path / "prompt.txt"),
        duration_seconds=10.0 if resolved else 20.0,
        telemetry=TelemetrySummary(total_tokens=100 if resolved else 200, total_steps=2, tool_calls=1),
        grade=GradeResult(resolved=resolved, status="graded"),
    )


def test_reporting_summarizes_and_writes(tmp_path: Path) -> None:
    records = [
        make_record(tmp_path, ArmName.MAS_CENTRALIZE, True, 0),
        make_record(tmp_path, ArmName.MAS_DECENTRALIZED, False, 0),
    ]
    summary = summarize(records)
    assert len(summary) == 2
    deltas = pairwise_deltas(records)
    assert deltas[0]["resolved_delta"] == -1
    report_dir = write_report(records, output_root=tmp_path, run_id="run-1")
    assert (report_dir / "report.md").exists()


def test_reporting_excludes_failed_grading_from_graded_counts(tmp_path: Path) -> None:
    records = [
        make_record(tmp_path, ArmName.MAS_CENTRALIZE, True, 0),
        RunRecord(
            run_id="run-1",
            arm=ArmName.MAS_DECENTRALIZED,
            instance_id="repo__issue-1",
            repeat_index=0,
            status=RunStatus.SUCCESS,
            workspace_dir=str(tmp_path / "workspace"),
            run_dir=str(tmp_path / "mas_decentralized"),
            prompt_path=str(tmp_path / "prompt.txt"),
            duration_seconds=20.0,
            telemetry=TelemetrySummary(total_tokens=200, total_steps=2, tool_calls=1),
            grade=GradeResult(resolved=False, status="grading_failed"),
        ),
    ]
    summary = summarize(records)
    by_arm = {row.arm: row for row in summary}
    assert by_arm[ArmName.MAS_CENTRALIZE].graded_runs == 1
    assert by_arm[ArmName.MAS_DECENTRALIZED].graded_runs == 0
    assert by_arm[ArmName.MAS_DECENTRALIZED].resolved_runs == 0
