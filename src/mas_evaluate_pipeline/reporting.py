"""Grade aggregation and reporting."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .constants import DEFAULT_ARMS
from .constants import DEFAULT_REPORTS_DIR
from .models import ArmName, ReportRow, RunRecord


def load_run_records(runs_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for path in sorted(runs_root.rglob("run_record.json")):
        records.append(RunRecord.model_validate_json(path.read_text(encoding="utf-8")))
    return records


def summarize(records: list[RunRecord]) -> list[ReportRow]:
    buckets: dict[ArmName, list[RunRecord]] = defaultdict(list)
    for record in records:
        buckets[record.arm].append(record)
    rows: list[ReportRow] = []
    for arm in sorted(buckets, key=lambda item: item.value):
        arm_records = buckets[arm]
        graded = [record for record in arm_records if _is_successfully_graded(record)]
        resolved = [record for record in graded if record.grade.resolved]
        rows.append(
            ReportRow(
                arm=arm,
                runs=len(arm_records),
                graded_runs=len(graded),
                resolved_runs=len(resolved),
                resolved_rate=(len(resolved) / len(graded)) if graded else 0.0,
                avg_duration_seconds=_avg(record.duration_seconds for record in arm_records),
                avg_total_tokens=_avg(record.telemetry.total_tokens for record in arm_records),
                avg_steps=_avg(record.telemetry.total_steps for record in arm_records),
                avg_tool_calls=_avg(record.telemetry.tool_calls for record in arm_records),
                avg_tool_failures=_avg(record.telemetry.tool_failures for record in arm_records),
            )
        )
    return rows


def pairwise_deltas(records: list[RunRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[ArmName, RunRecord]] = defaultdict(dict)
    for record in records:
        grouped[(record.instance_id, record.repeat_index)][record.arm] = record
    deltas: list[dict[str, object]] = []
    for (instance_id, repeat_index), arm_records in sorted(grouped.items()):
        if len(arm_records) < 2:
            continue
        ordered = sorted(arm_records.items(), key=lambda item: DEFAULT_ARMS.index(item[0].value))
        base_arm, base_record = ordered[0]
        for arm, record in ordered[1:]:
            deltas.append(
                {
                    "instance_id": instance_id,
                    "repeat_index": repeat_index,
                    "base_arm": base_arm.value,
                    "compare_arm": arm.value,
                    "resolved_delta": int(bool(record.grade.resolved)) - int(bool(base_record.grade.resolved)),
                    "token_delta": record.telemetry.total_tokens - base_record.telemetry.total_tokens,
                    "latency_delta": record.duration_seconds - base_record.duration_seconds,
                    "step_delta": record.telemetry.total_steps - base_record.telemetry.total_steps,
                }
            )
    return deltas


def write_report(records: list[RunRecord], *, output_root: Path, run_id: str) -> Path:
    report_dir = output_root / DEFAULT_REPORTS_DIR / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(records)
    delta_rows = pairwise_deltas(records)
    (report_dir / "summary.json").write_text(
        json.dumps([row.model_dump() for row in summary_rows], indent=2, default=str),
        encoding="utf-8",
    )
    (report_dir / "paired_deltas.json").write_text(
        json.dumps(delta_rows, indent=2, default=str),
        encoding="utf-8",
    )
    _write_summary_csv(summary_rows, report_dir / "summary.csv")
    markdown = _render_markdown(summary_rows, delta_rows)
    (report_dir / "report.md").write_text(markdown, encoding="utf-8")
    return report_dir


def _write_summary_csv(rows: list[ReportRow], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].model_dump().keys()) if rows else ["arm"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def _render_markdown(summary_rows: list[ReportRow], delta_rows: list[dict[str, object]]) -> str:
    lines = [
        "# SWE-bench Communication Benchmark Report",
        "",
        "## Summary",
        "",
        "| Arm | Runs | Graded | Resolved | Resolved Rate | Avg Duration (s) | Avg Tokens | Avg Steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.arm.value} | {row.runs} | {row.graded_runs} | {row.resolved_runs} | "
            f"{row.resolved_rate:.2%} | {row.avg_duration_seconds:.2f} | {row.avg_total_tokens:.2f} | {row.avg_steps:.2f} |"
        )
    lines.extend(["", "## Paired Deltas", "", f"Compared pairs: {len(delta_rows)}"])
    return "\n".join(lines) + "\n"


def _avg(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _is_successfully_graded(record: RunRecord) -> bool:
    return record.grade.status == "graded" and record.grade.resolved is not None
