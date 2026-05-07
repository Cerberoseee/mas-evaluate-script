from __future__ import annotations

from pathlib import Path

from mas_evaluate_pipeline.manifest import build_task_manifest, load_dataset_records_from_jsonl, load_manifest, save_manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    records = load_dataset_records_from_jsonl(Path("tests/data/sample_tasks.jsonl"))
    manifest = build_task_manifest(records, dataset_name="SWE-bench/SWE-bench_Lite", dataset_split="test", count=1)
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].instance_id == manifest.tasks[0].instance_id
