"""Task manifest helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .constants import DEFAULT_REPO_ORDER, DEFAULT_TASK_COUNT
from .models import TaskInstance, TaskManifest


def load_dataset_records_from_jsonl(path: Path) -> list[TaskInstance]:
    tasks: list[TaskInstance] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            tasks.append(_task_from_mapping(raw))
    return tasks


def load_dataset_records(dataset_name: str, split: str) -> list[TaskInstance]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via fallback in tests
        raise RuntimeError(
            "The 'datasets' package is not installed. Pass --dataset-jsonl or install datasets."
        ) from exc
    dataset = load_dataset(dataset_name, split=split)
    return [_task_from_mapping(item) for item in dataset]


def build_task_manifest(
    records: list[TaskInstance],
    *,
    dataset_name: str,
    dataset_split: str,
    count: int = DEFAULT_TASK_COUNT,
    explicit_ids: list[str] | None = None,
) -> TaskManifest:
    selected = select_curated_subset(records, count=count, explicit_ids=explicit_ids)
    strategy = "explicit_ids" if explicit_ids else "repo_balanced_round_robin"
    return TaskManifest(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        selection_strategy=strategy,
        tasks=selected,
    )


def save_manifest(manifest: TaskManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(path: Path) -> TaskManifest:
    return TaskManifest.model_validate_json(path.read_text(encoding="utf-8"))


def select_curated_subset(
    records: list[TaskInstance],
    *,
    count: int,
    explicit_ids: list[str] | None = None,
) -> list[TaskInstance]:
    by_id = {task.instance_id: task for task in records}
    if explicit_ids:
        missing = [instance_id for instance_id in explicit_ids if instance_id not in by_id]
        if missing:
            raise ValueError(f"Unknown instance ids: {missing}")
        return [by_id[instance_id] for instance_id in explicit_ids][:count]

    grouped: dict[str, list[TaskInstance]] = defaultdict(list)
    for task in sorted(records, key=lambda item: (repo_rank(item.repo), item.instance_id)):
        grouped[task.repo].append(task)

    ordered_repos = sorted(grouped, key=repo_rank)
    selected: list[TaskInstance] = []
    index = 0
    while len(selected) < count and ordered_repos:
        repo = ordered_repos[index % len(ordered_repos)]
        if grouped[repo]:
            selected.append(grouped[repo].pop(0))
        if not grouped[repo]:
            ordered_repos.remove(repo)
            if not ordered_repos:
                break
            index %= len(ordered_repos)
            continue
        index += 1
    return selected


def repo_rank(repo: str) -> tuple[int, str]:
    try:
        return (DEFAULT_REPO_ORDER.index(repo), repo)
    except ValueError:
        return (len(DEFAULT_REPO_ORDER), repo)


def instance_ids(tasks: Iterable[TaskInstance]) -> list[str]:
    return [task.instance_id for task in tasks]


def _task_from_mapping(raw: dict) -> TaskInstance:
    return TaskInstance(
        instance_id=raw["instance_id"],
        repo=raw["repo"],
        base_commit=raw.get("base_commit"),
        problem_statement=raw["problem_statement"],
        version=raw.get("version"),
        hints_text=raw.get("hints_text"),
        fail_to_pass=_coerce_test_list(raw.get("FAIL_TO_PASS") or raw.get("fail_to_pass")),
        pass_to_pass=_coerce_test_list(raw.get("PASS_TO_PASS") or raw.get("pass_to_pass")),
        environment_setup_commit=raw.get("environment_setup_commit"),
        metadata={
            key: value
            for key, value in raw.items()
            if key
            not in {
                "instance_id",
                "repo",
                "base_commit",
                "problem_statement",
                "version",
                "hints_text",
                "FAIL_TO_PASS",
                "fail_to_pass",
                "PASS_TO_PASS",
                "pass_to_pass",
                "environment_setup_commit",
            }
        },
    )


def _coerce_test_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]
