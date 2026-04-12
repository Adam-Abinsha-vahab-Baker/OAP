import pytest
import threading
from tests.fake_agent_server import FakeAgentHandler, HTTPServer
from oap.envelope import TaskEnvelope
from oap.adapters.http import HTTPAdapter
from oap.router import OAPRouter


PORT = 19999  # high port, avoids permission issues


@pytest.fixture(scope="module")
def fake_server():
    """Spin up the fake agent server in a background thread for the test module."""
    server = HTTPServer(("localhost", PORT), FakeAgentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://localhost:{PORT}"
    server.shutdown()


async def test_http_adapter_invoke(fake_server):
    adapter = HTTPAdapter(agent_id="fake-agent", base_url=fake_server)
    envelope = TaskEnvelope(goal="find the best open source LLMs")

    agent_input = adapter.to_agent_format(envelope)
    raw_output = await adapter.invoke(agent_input)

    assert "result" in raw_output
    assert "find the best open source LLMs" in raw_output["result"]


async def test_http_adapter_updates_memory(fake_server):
    adapter = HTTPAdapter(agent_id="fake-agent", base_url=fake_server)
    envelope = TaskEnvelope(goal="research vector databases")

    agent_input = adapter.to_agent_format(envelope)
    raw_output = await adapter.invoke(agent_input)
    result = adapter.to_envelope(raw_output, envelope)

    assert result.memory["last_result"] is not None
    assert result.memory["processed_by"] == "fake-agent"


async def test_http_adapter_preserves_goal(fake_server):
    adapter = HTTPAdapter(agent_id="fake-agent", base_url=fake_server)
    envelope = TaskEnvelope(goal="debug my python script")

    agent_input = adapter.to_agent_format(envelope)
    raw_output = await adapter.invoke(agent_input)
    result = adapter.to_envelope(raw_output, envelope)

    assert result.goal == envelope.goal
    assert result.id == envelope.id


async def test_router_with_http_adapter(fake_server):
    router = OAPRouter()
    router.register(
        "fake-agent",
        HTTPAdapter(agent_id="fake-agent", base_url=fake_server),
        capabilities=["research", "find", "search"],
    )

    envelope = TaskEnvelope(goal="research the best vector databases")
    result = await router.route(envelope)

    assert len(result.steps_taken) == 1
    assert result.steps_taken[0].agent_id == "fake-agent"
    assert "last_result" in result.memory