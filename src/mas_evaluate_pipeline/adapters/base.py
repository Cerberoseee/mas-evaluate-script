"""Adapter protocol and shared helpers."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

from ..models import ArmName, RunArtifacts, StudyConfig, TaskInstance


class BenchmarkAdapter(ABC):
    arm: ArmName

    def __init__(self, study_config: StudyConfig) -> None:
        self.study_config = study_config

    @abstractmethod
    def run(
        self,
        *,
        task: TaskInstance,
        prompt: str,
        run_dir: Path,
        workspace_dir: Path,
        repeat_index: int,
    ) -> RunArtifacts:
        """Execute one arm against one prepared task."""


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
