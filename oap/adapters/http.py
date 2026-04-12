from __future__ import annotations
from typing import Any
from oap.adapters.base import AgentAdapter
from oap.envelope import TaskEnvelope
from oap.transport.http import HTTPTransport


class HTTPAdapter(AgentAdapter):
    """Adapter for any agent that exposes a POST /invoke endpoint."""

    def __init__(self, agent_id: str, base_url: str, timeout: float = 30.0):
        self.agent_id = agent_id
        self.transport = HTTPTransport(base_url=base_url, timeout=timeout)

    def to_agent_format(self, envelope: TaskEnvelope) -> TaskEnvelope:
        return envelope

    async def invoke(self, agent_input: Any) -> dict:
        return await self.transport.invoke(agent_input)

    def to_envelope(self, agent_output: dict, previous: TaskEnvelope) -> TaskEnvelope:
        updated = previous.model_copy(deep=True)

        if "memory" in agent_output:
            updated.memory.update(agent_output["memory"])

        if "result" in agent_output:
            updated.memory["last_result"] = agent_output["result"]

        if "handoff" in agent_output and agent_output["handoff"]:
            h = agent_output["handoff"]
            updated.with_handoff(
                next_agent=h["next_agent"],
                reason=h.get("reason", ""),
                partial_result=h.get("partial_result"),
            )

        return updated