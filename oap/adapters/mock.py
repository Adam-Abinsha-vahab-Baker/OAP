from __future__ import annotations
from typing import Any
from oap.adapters.base import AgentAdapter
from oap.envelope import TaskEnvelope


class MockAgentAdapter(AgentAdapter):
    """A fake agent that returns a canned response. Used for testing."""

    def __init__(
        self,
        agent_id: str = "mock",
        response: str = "Mock result",
        next_agent: str | None = None,
    ):
        self.agent_id = agent_id
        self.response = response
        self.next_agent = next_agent

    def to_agent_format(self, envelope: TaskEnvelope) -> dict:
        return {"goal": envelope.goal, "memory": envelope.memory}

    async def invoke(self, agent_input: Any) -> dict:
        out: dict[str, Any] = {"result": self.response, "goal": agent_input["goal"]}
        if self.next_agent:
            out["handoff"] = {"next_agent": self.next_agent, "reason": "mock handoff"}
        return out

    def to_envelope(self, agent_output: Any, previous: TaskEnvelope) -> TaskEnvelope:
        updated = previous.model_copy(deep=True)
        updated.memory["last_result"] = agent_output.get("result")
        if agent_output.get("handoff"):
            h = agent_output["handoff"]
            updated.with_handoff(
                next_agent=h["next_agent"],
                reason=h.get("reason", ""),
                partial_result=h.get("partial_result"),
            )
        else:
            updated.handoff = None
        return updated
MockAdapter = MockAgentAdapter
