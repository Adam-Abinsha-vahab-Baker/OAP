import pytest
from unittest.mock import AsyncMock, MagicMock
from oap.router import OAPRouter, RoutingError
from oap.envelope import TaskEnvelope
from oap.adapters.mock import MockAdapter


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.is_available.return_value = True
    provider.complete = AsyncMock(return_value="research-agent")
    return provider


@pytest.fixture
def router_with_llm(mock_provider):
    router = OAPRouter(llm_provider=mock_provider)
    router.register("research-agent", MockAdapter(), ["research"], description="Researches topics")
    router.register("summarise-agent", MockAdapter(), ["summarise"], description="Summarises text")
    return router


@pytest.mark.asyncio
async def test_llm_routing_used(router_with_llm, mock_provider):
    envelope = TaskEnvelope(goal="I need help understanding attention mechanisms")
    agent_id = await router_with_llm.select_agent(envelope)
    assert agent_id == "research-agent"
    mock_provider.complete.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_on_unknown_llm_response(router_with_llm, mock_provider):
    mock_provider.complete = AsyncMock(return_value="nonexistent-agent")
    envelope = TaskEnvelope(goal="research something")
    # Falls back to keyword match — "research" matches research-agent
    agent_id = await router_with_llm.select_agent(envelope)
    assert agent_id == "research-agent"


@pytest.mark.asyncio
async def test_no_match_from_llm_raises(router_with_llm, mock_provider):
    mock_provider.complete = AsyncMock(return_value="NO_MATCH")
    envelope = TaskEnvelope(goal="something nobody can handle")
    with pytest.raises(RoutingError):
        await router_with_llm.select_agent(envelope)


@pytest.mark.asyncio
async def test_explicit_handoff_overrides_llm(router_with_llm, mock_provider):
    envelope = TaskEnvelope(goal="anything")
    envelope.with_handoff("summarise-agent", "explicit")
    agent_id = await router_with_llm.select_agent(envelope)
    assert agent_id == "summarise-agent"
    mock_provider.complete.assert_not_called()


@pytest.mark.asyncio
async def test_no_provider_uses_keyword_matching():
    router = OAPRouter()
    router.register("research-agent", MockAdapter(), ["research"])
    envelope = TaskEnvelope(goal="research something")
    agent_id = await router.select_agent(envelope)
    assert agent_id == "research-agent"


@pytest.mark.asyncio
async def test_llm_exception_falls_back(router_with_llm, mock_provider):
    mock_provider.complete = AsyncMock(side_effect=Exception("API down"))
    envelope = TaskEnvelope(goal="research something")
    agent_id = await router_with_llm.select_agent(envelope)
    assert agent_id == "research-agent"
