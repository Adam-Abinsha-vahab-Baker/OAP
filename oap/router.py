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

    async def chain(
        self,
        envelope: TaskEnvelope,
        max_hops: int = 10,
        on_hop: object = None,
    ) -> tuple[TaskEnvelope, list[str]]:
        """Route repeatedly, following handoffs until none remains or max_hops is reached.

        Args:
            envelope: The starting TaskEnvelope.
            max_hops: Hard ceiling on the number of agent calls.
            on_hop: Optional callable(hop_number, agent_id) invoked after each hop.

        Returns:
            A tuple of (final_envelope, list_of_agent_ids_visited).
        """
        current = envelope
        visited: list[str] = []

        for hop in range(1, max_hops + 1):
            agent_id = self.select_agent(current)
            current = await self.route(current)
            visited.append(agent_id)

            if on_hop:
                on_hop(hop, agent_id)  # type: ignore[operator]

            if not current.handoff:
                break
        else:
            # Exited loop because max_hops was reached without handoff clearing
            pass

        return current, visited

    async def run_pipeline(
        self,
        envelope: TaskEnvelope,
        agent_ids: list[str],
        on_hop: object = None,
    ) -> tuple[TaskEnvelope, list[str]]:
        """Route through a fixed ordered list of agents, ignoring capability matching and handoffs.

        Args:
            envelope: The starting TaskEnvelope.
            agent_ids: Ordered list of agent IDs to invoke in sequence.
            on_hop: Optional callable(hop_number, total, agent_id) invoked after each hop.

        Raises:
            RoutingError: If any agent_id is not registered.

        Returns:
            A tuple of (final_envelope, agent_ids).
        """
        missing = [aid for aid in agent_ids if aid not in self._agents]
        if missing:
            raise RoutingError(
                f"Pipeline agent(s) not found in registry: {', '.join(missing)}"
            )

        current = envelope
        total = len(agent_ids)

        for hop, agent_id in enumerate(agent_ids, 1):
            adapter = self._agents[agent_id]
            agent_input = adapter.to_agent_format(current)
            agent_output = await adapter.invoke(agent_input)
            current = adapter.to_envelope(agent_output, current)
            current.handoff = None  # pipeline ignores handoffs
            current.add_step(
                agent_id=agent_id,
                action=f"Pipeline hop {hop}/{total}: {agent_id}",
                result=agent_output,
            )

            if on_hop:
                on_hop(hop, total, agent_id)  # type: ignore[operator]

        return current, agent_ids