import pytest
from oap.envelope import TaskEnvelope
from oap.router import OAPRouter, RoutingError
from oap.adapters.mock import MockAgentAdapter


def make_router() -> OAPRouter:
    router = OAPRouter()
    router.register(
        "research-agent",
        MockAgentAdapter("research-agent", response="research done"),
        capabilities=["research", "find", "search"],
    )
    router.register(
        "coding-agent",
        MockAgentAdapter("coding-agent", response="code written"),
        capabilities=["code", "implement", "debug"],
    )
    return router


def test_list_agents():
    router = make_router()
    agents = router.list_agents()
    ids = [a["id"] for a in agents]
    assert "research-agent" in ids
    assert "coding-agent" in ids


async def test_capability_match_research():
    router = make_router()
    e = TaskEnvelope(goal="research the best vector databases")
    assert await router.select_agent(e) == "research-agent"


async def test_capability_match_coding():
    router = make_router()
    e = TaskEnvelope(goal="implement a quick sort algorithm")
    assert await router.select_agent(e) == "coding-agent"


async def test_explicit_handoff_overrides_capability():
    router = make_router()
    e = TaskEnvelope(goal="research something")
    e.with_handoff(next_agent="coding-agent", reason="user override")
    assert await router.select_agent(e) == "coding-agent"


async def test_no_match_raises_routing_error():
    router = make_router()
    e = TaskEnvelope(goal="make me a sandwich")
    with pytest.raises(RoutingError):
        await router.select_agent(e)


async def test_empty_router_raises_routing_error():
    router = OAPRouter()
    e = TaskEnvelope(goal="research something")
    with pytest.raises(RoutingError, match="No agents are registered"):
        await router.select_agent(e)


async def test_unknown_handoff_agent_raises_routing_error():
    router = make_router()
    e = TaskEnvelope(goal="do something")
    e.with_handoff(next_agent="nonexistent-agent", reason="test")
    with pytest.raises(RoutingError):
        await router.select_agent(e)


async def test_route_updates_memory():
    router = make_router()
    e = TaskEnvelope(goal="research neural networks")
    result = await router.route(e)
    assert "last_result" in result.memory
    assert result.memory["last_result"] == "research done"


async def test_route_appends_step():
    router = make_router()
    e = TaskEnvelope(goal="debug my code")
    result = await router.route(e)
    assert len(result.steps_taken) == 1
    assert result.steps_taken[0].agent_id == "coding-agent"


async def test_route_preserves_goal():
    router = make_router()
    e = TaskEnvelope(goal="find open source llm frameworks")
    result = await router.route(e)
    assert result.goal == e.goal
    assert result.id == e.id
