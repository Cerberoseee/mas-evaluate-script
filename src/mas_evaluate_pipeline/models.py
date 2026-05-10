"""Pydantic models for study configuration and artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .constants import (
    DEFAULT_ARMS,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_GRADE_WORK_DIR,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPEATS,
    DEFAULT_REPO_CACHE_DIR,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOTAL_TOKEN_BUDGET,
)


class ArmName(str, Enum):
    MINI_SWE_AGENT = "mini_swe_agent"
    MAS_CENTRALIZE = "mas_centralize"
    MAS_DECENTRALIZED = "mas_decentralized"


class FailureClass(str, Enum):
    NONE = "none"
    HARNESS_SETUP = "harness_setup_failure"
    REPOSITORY_PREPARATION = "repository_preparation_failure"
    ADAPTER_RUNTIME = "adapter_runtime_failure"
    TIMEOUT = "timeout"
    NO_PATCH = "no_patch_or_invalid_patch"
    GRADED_FAIL = "graded_fail"


class RunStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskInstance(BaseModel):
    instance_id: str
    repo: str
    base_commit: str | None = None
    problem_statement: str
    version: str | None = None
    hints_text: str | None = None
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    environment_setup_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskManifest(BaseModel):
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_split: str = DEFAULT_DATASET_SPLIT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    selection_strategy: str
    tasks: list[TaskInstance]


class ShellPolicy(BaseModel):
    allowlist: list[str] = Field(
        default_factory=lambda: [
            "python",
            "python3",
            "pytest",
            "py.test",
            "uv",
            "pip",
            "pip3",
            "tox",
            "git",
            "ls",
            "cat",
            "sed",
            "grep",
            "rg",
        ]
    )
    max_command_seconds: int = 300


class AdapterConfig(BaseModel):
    command: list[str] | None = None
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    prompt_preamble: str = ""
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    shell_policy: ShellPolicy = Field(default_factory=ShellPolicy)
    openai_api_key: str | None = None


class HarnessConfig(BaseModel):
    dataset_name: str = DEFAULT_DATASET_NAME
    dataset_split: str = DEFAULT_DATASET_SPLIT
    repo_cache_dir: str = str(DEFAULT_REPO_CACHE_DIR)
    grade_work_dir: str = str(DEFAULT_GRADE_WORK_DIR)
    namespace: str | None = "none"
    instance_image_tag: str = "latest"
    env_image_tag: str = "latest"
    repo_mirrors: dict[str, str] = Field(default_factory=dict)
    grade_command: list[str] | None = None
    prepare_command: list[str] | None = None
    extra_env: dict[str, str] = Field(default_factory=dict)
    # Docker-based execution: pull the official SWE-bench per-instance image and
    # run the agent inside the container rather than on the host.  When True,
    # host-side workspace cloning and venv provisioning are skipped entirely —
    # the container already has the repo at base_commit and all C extensions
    # compiled against the original build toolchain.
    use_docker: bool = False
    # Prefix of the SWE-bench per-instance Docker image name.
    # Full image = f"{docker_image_prefix}.{instance_id}:{instance_image_tag}"
    docker_image_prefix: str = "swebench/sweb.eval.x86_64"
    # Runtime pre-provisioning — creates workspace/venv/ before the agent starts.
    # Uses uv to avoid PEP 668 and to support per-repo Python versions.
    # Ignored when use_docker=True (the container provides the environment).
    auto_provision_runtime: bool = True
    # Default Python version for provisioned venvs; overridden per-repo via provision_python_versions.
    # 3.9 is a good baseline: it is managed by uv on this host and NumPy 1.24.x ships
    # Python 3.9 wheels, satisfying the numpy<2 constraint for 2022-era repos like astropy.
    provision_python_version: str = "3.9"
    # Per-repo overrides, keyed by "org/repo" (e.g. "astropy/astropy").
    provision_python_versions: dict[str, str] = Field(default_factory=dict)
    # Extra packages to install in every provisioned venv (appended to the built-in common set).
    provision_extra_packages: list[str] = Field(default_factory=list)


class StudyConfig(BaseModel):
    run_id: str | None = None
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    env_file: str | None = ".env"
    openai_api_key: str | None = None
    base_model: str = "gpt-4o-mini"
    total_token_budget: int = DEFAULT_TOTAL_TOKEN_BUDGET
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    repeats: int = DEFAULT_REPEATS
    arms: list[ArmName] = Field(default_factory=lambda: [ArmName(value) for value in DEFAULT_ARMS])
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    adapters: dict[str, AdapterConfig] = Field(default_factory=dict)

    @field_validator("arms", mode="before")
    @classmethod
    def _coerce_arms(cls, value: Any) -> list[ArmName]:
        if value is None:
            return [ArmName(item) for item in DEFAULT_ARMS]
        return [item if isinstance(item, ArmName) else ArmName(item) for item in value]

    def adapter_config_for(self, arm: ArmName) -> AdapterConfig:
        return self.adapters.get(arm.value, AdapterConfig())


class TelemetrySummary(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_steps: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    retries: int = 0
    escalations: int = 0
    messages: int = 0
    handoffs: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class GradeResult(BaseModel):
    resolved: bool | None = None
    status: str = "not_graded"
    details: dict[str, Any] = Field(default_factory=dict)
    graded_at: datetime | None = None


class RunRecord(BaseModel):
    run_id: str
    arm: ArmName
    instance_id: str
    repeat_index: int
    status: RunStatus = RunStatus.PENDING
    failure_class: FailureClass = FailureClass.NONE
    note: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    workspace_dir: str
    run_dir: str
    prompt_path: str
    patch_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    telemetry_path: str | None = None
    result_path: str | None = None
    grade_path: str | None = None
    prompt_sha256: str = ""
    telemetry: TelemetrySummary = Field(default_factory=TelemetrySummary)
    grade: GradeResult = Field(default_factory=GradeResult)


class RunArtifacts(BaseModel):
    status: RunStatus
    failure_class: FailureClass = FailureClass.NONE
    note: str = ""
    patch_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    telemetry_path: str | None = None
    result_path: str | None = None
    telemetry: TelemetrySummary = Field(default_factory=TelemetrySummary)


class PreparedTask(BaseModel):
    task: TaskInstance
    workspace_dir: str
    metadata_path: str
    source_repo_dir: str | None = None


class ReportRow(BaseModel):
    arm: ArmName
    runs: int
    graded_runs: int
    resolved_runs: int
    resolved_rate: float
    avg_duration_seconds: float
    avg_total_tokens: float
    avg_steps: float
    avg_tool_calls: float
    avg_tool_failures: float


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)
