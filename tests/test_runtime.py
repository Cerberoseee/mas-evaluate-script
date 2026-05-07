from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mas_evaluate_pipeline.manifest import build_task_manifest, load_dataset_records_from_jsonl
from mas_evaluate_pipeline.models import StudyConfig
from mas_evaluate_pipeline.runtime import BenchmarkRuntime


def init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "foo.py").write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True, text=True)


def test_runtime_runs_all_arms_with_fake_runner(tmp_path: Path) -> None:
    repo_dir = tmp_path / "mirror"
    init_repo(repo_dir)
    config = StudyConfig.model_validate(
        {
            "run_id": "test-run",
            "output_root": str(tmp_path),
            "repeats": 1,
            "arms": ["mas_centralize", "mas_decentralized"],
            "harness": {"repo_mirrors": {"example/repo": str(repo_dir)}},
            "adapters": {
                arm: {
                    "command": [sys.executable, "tests/fixtures/fake_adapter_runner.py"],
                }
                for arm in ["mas_centralize", "mas_decentralized"]
            },
        }
    )
    records_source = load_dataset_records_from_jsonl(Path("tests/data/sample_tasks.jsonl"))
    manifest = build_task_manifest(records_source, dataset_name="SWE-bench/SWE-bench_Lite", dataset_split="test", count=1)
    runtime = BenchmarkRuntime(config)
    records = runtime.run_manifest(manifest, output_root=tmp_path)
    assert len(records) == 2
    assert all(record.status.value == "success" for record in records)
    assert all(Path(record.patch_path).exists() for record in records if record.patch_path)
