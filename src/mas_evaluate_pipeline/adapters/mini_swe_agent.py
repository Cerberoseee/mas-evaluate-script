"""mini-swe-agent wrapper adapter."""

from __future__ import annotations

from ..models import AdapterConfig, ArmName
from .command import CommandAdapter


class MiniSweAgentAdapter(CommandAdapter):
    arm = ArmName.MINI_SWE_AGENT

    def __init__(self, study_config, adapter_config: AdapterConfig) -> None:
        super().__init__(study_config, adapter_config)
