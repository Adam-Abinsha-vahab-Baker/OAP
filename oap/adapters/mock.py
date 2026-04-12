from __future__ import annotations
from typing import Any
from oap.adapters.base import AgentAdapter
from oap.envelope import TaskEnvelope


class MockAgentAdapter(AgentAdapter):
    """A fake agent that returns a canned response. Used for testing."""

    def __init__(self, agent_id: str, response: str = "Mock result"):
        self.agent_id = agent_id
        self.response = response

    def to_agent_format(self, envelope: TaskEnvelope) -> dict:
        return {"goal": envelope.goal, "memory": envelope.memory}

    async def invoke(self, agent_input: Any) -> dict:
        return {"result": self.response, "goal": agent_input["goal"]}

    def to_envelope(self, agent_output: Any, previous: TaskEnvelope) -> TaskEnvelope:
        updated = previous.model_copy(deep=True)
        updated.memory["last_result"] = agent_output.get("result")
        return updated