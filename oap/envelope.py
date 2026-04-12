from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field


class Step(BaseModel):
    agent_id: str
    action: str
    result: Any = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Constraints(BaseModel):
    max_cost_usd: float | None = None
    allowed_tools: list[str] | None = None
    deadline_ms: int | None = None


class Handoff(BaseModel):
    next_agent: str
    reason: str
    partial_result: Any = None


class TaskEnvelope(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    version: str = "0.1"
    goal: str
    memory: dict[str, Any] = Field(default_factory=dict)
    steps_taken: list[Step] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    handoff: Handoff | None = None

    def add_step(self, agent_id: str, action: str, result: Any = None) -> None:
        self.steps_taken.append(Step(agent_id=agent_id, action=action, result=result))

    def with_handoff(self, next_agent: str, reason: str, partial_result: Any = None) -> "TaskEnvelope":
        self.handoff = Handoff(next_agent=next_agent, reason=reason, partial_result=partial_result)
        return self