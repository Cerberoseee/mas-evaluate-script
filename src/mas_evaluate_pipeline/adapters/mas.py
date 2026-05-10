"""MAS system command adapters."""

from __future__ import annotations

from ..models import AdapterConfig, ArmName
from .command import CommandAdapter


class MasCentralizeAdapter(CommandAdapter):
    arm = ArmName.MAS_CENTRALIZE
    # The AutoGen orchestrator and MCP servers run on the host; only the
    # Engineer's bash commands are routed into the SWE-bench Docker container.
    _bash_via_docker_env = True

    def __init__(self, study_config, adapter_config: AdapterConfig) -> None:
        super().__init__(study_config, adapter_config)


class MasDecentralizedAdapter(CommandAdapter):
    arm = ArmName.MAS_DECENTRALIZED

    def __init__(self, study_config, adapter_config: AdapterConfig) -> None:
        super().__init__(study_config, adapter_config)
