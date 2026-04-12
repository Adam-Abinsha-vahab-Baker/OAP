from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Any
from oap.envelope import TaskEnvelope
from oap.adapters.base import AgentAdapter


class RoutingError(Exception):
    pass


class OAPRouter:
    def __init__(self):
        self._agents: dict[str, AgentAdapter] = {}
        self._capabilities: dict[str, list[str]] = {}

    def register(self, agent_id: str, adapter: AgentAdapter, capabilities: list[str]) -> None:
        """Register an agent with its capabilities."""
        self._agents[agent_id] = adapter
        self._capabilities[agent_id] = [c.lower() for c in capabilities]

    def list_agents(self) -> list[dict[str, Any]]:
        """Return all registered agents and their capabilities."""
        return [
            {"id": agent_id, "capabilities": caps}
            for agent_id, caps in self._capabilities.items()
        ]

    def select_agent(self, envelope: TaskEnvelope) -> str:
        """Pick the best agent for this envelope.

        Priority:
        1. Explicit handoff.next_agent in the envelope
        2. Capability match against goal keywords (most matches wins)
        3. Raise RoutingError if nothing matches or there is a tie
        """
        if envelope.handoff and envelope.handoff.next_agent:
            agent_id = envelope.handoff.next_agent
            if agent_id not in self._agents:
                raise RoutingError(f"Requested agent '{agent_id}' is not registered.")
            return agent_id

        return self._match_by_capability(envelope.goal)

    def _match_by_capability(self, goal: str) -> str:
        """Score each agent by how many capability keywords appear in the goal."""
        goal_lower = goal.lower()
        scores: dict[str, int] = {}

        for agent_id, caps in self._capabilities.items():
            score = sum(
                1 for cap in caps
                if re.search(rf"\b{re.escape(cap)}\b", goal_lower)
            )
            scores[agent_id] = score

        if not scores:
            raise RoutingError("No agents are registered.")

        best_score = max(scores.values())

        if best_score == 0:
            raise RoutingError(
                f"No agent matched the goal: '{goal}'\n"
                f"Registered capabilities: {self._capabilities}"
            )

        winners = [aid for aid, score in scores.items() if score == best_score]
        if len(winners) > 1:
            raise RoutingError(
                f"Ambiguous routing for goal: '{goal}'\n"
                f"Tied agents: {winners} — add a handoff.next_agent to disambiguate."
            )

        return winners[0]

    async def route(self, envelope: TaskEnvelope) -> TaskEnvelope:
        """Route an envelope to the correct agent and return the updated envelope."""
        agent_id = self.select_agent(envelope)
        adapter = self._agents[agent_id]

        agent_input = adapter.to_agent_format(envelope)
        agent_output = await adapter.invoke(agent_input)
        result = adapter.to_envelope(agent_output, envelope)

        result.add_step(
            agent_id=agent_id,
            action=f"Routed to {agent_id}",
            result=agent_output,
        )
        return result