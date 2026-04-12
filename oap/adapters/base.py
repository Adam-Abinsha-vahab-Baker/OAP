from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from oap.envelope import TaskEnvelope


class AgentAdapter(ABC):

    @abstractmethod
    def to_agent_format(self, envelope: TaskEnvelope) -> Any:
        """Translate a TaskEnvelope into whatever format this agent expects."""
        ...

    @abstractmethod
    async def invoke(self, agent_input: Any) -> Any:
        """Call the agent and return its raw output."""
        ...

    @abstractmethod
    def to_envelope(self, agent_output: Any, previous: TaskEnvelope) -> TaskEnvelope:
        """Translate the agent's output back into a TaskEnvelope."""
        ...