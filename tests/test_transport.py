"""Tests for HTTPTransport retry/backoff logic and oap ping CLI command."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
import respx
from typer.testing import CliRunner

from oap.cli import app
from oap.envelope import TaskEnvelope
from oap.router import RoutingError
from oap.transport.http import HTTPTransport
import oap.registry as reg

runner = CliRunner()
_PING_PORT = 19996


# ---------------------------------------------------------------------------
# Fake server for ping tests
# ---------------------------------------------------------------------------

class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        body = json.dumps({"result": "ok", "memory": {}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def ping_server():
    server = HTTPServer(("localhost", _PING_PORT), PingHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://localhost:{_PING_PORT}"
    server.shutdown()


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY_PATH", tmp_path / "agents.json")


# ---------------------------------------------------------------------------
# HTTPTransport retry tests (using respx to mock httpx)
# ---------------------------------------------------------------------------

@respx.mock
async def test_retry_succeeds_on_second_attempt():
    """ReadTimeout on attempt 1, success on attempt 2."""
    envelope = TaskEnvelope(goal="test")
    transport = HTTPTransport(base_url="http://fake-agent", timeout=1.0)

    route = respx.post("http://fake-agent/invoke")
    route.side_effect = [
        httpx.ReadTimeout("timed out"),
        httpx.Response(200, json={"result": "ok", "memory": {}}),
    ]

    # Patch sleep so the test doesn't actually wait
    import oap.transport.http as http_mod
    original_sleep = http_mod.asyncio.sleep
    http_mod.asyncio.sleep = lambda _: original_sleep(0)

    result = await transport.invoke(envelope)
    assert result["result"] == "ok"
    assert route.call_count == 2

    http_mod.asyncio.sleep = original_sleep


@respx.mock
async def test_retry_exhausted_raises_routing_error():
    """All 4 attempts fail — raises RoutingError mentioning attempt count."""
    envelope = TaskEnvelope(goal="test")
    transport = HTTPTransport(base_url="http://fake-agent", timeout=1.0)

    respx.post("http://fake-agent/invoke").mock(
        side_effect=httpx.ReadTimeout("timed out")
    )

    import oap.transport.http as http_mod
    original_sleep = http_mod.asyncio.sleep
    http_mod.asyncio.sleep = lambda _: original_sleep(0)

    with pytest.raises(RoutingError) as exc_info:
        await transport.invoke(envelope)

    http_mod.asyncio.sleep = original_sleep

    assert "4 attempt(s)" in str(exc_info.value)
    assert "http://fake-agent" in str(exc_info.value)


@respx.mock
async def test_connect_error_triggers_retry():
    """ConnectError on attempt 1, success on attempt 2."""
    envelope = TaskEnvelope(goal="test")
    transport = HTTPTransport(base_url="http://fake-agent", timeout=1.0)

    route = respx.post("http://fake-agent/invoke")
    route.side_effect = [
        httpx.ConnectError("connection refused"),
        httpx.Response(200, json={"result": "ok", "memory": {}}),
    ]

    import oap.transport.http as http_mod
    original_sleep = http_mod.asyncio.sleep
    http_mod.asyncio.sleep = lambda _: original_sleep(0)

    result = await transport.invoke(envelope)
    assert result["result"] == "ok"
    assert route.call_count == 2

    http_mod.asyncio.sleep = original_sleep


@respx.mock
async def test_5xx_triggers_retry():
    """500 on attempt 1, success on attempt 2."""
    envelope = TaskEnvelope(goal="test")
    transport = HTTPTransport(base_url="http://fake-agent", timeout=1.0)

    route = respx.post("http://fake-agent/invoke")
    route.side_effect = [
        httpx.Response(500, text="internal error"),
        httpx.Response(200, json={"result": "ok", "memory": {}}),
    ]

    import oap.transport.http as http_mod
    original_sleep = http_mod.asyncio.sleep
    http_mod.asyncio.sleep = lambda _: original_sleep(0)

    result = await transport.invoke(envelope)
    assert result["result"] == "ok"
    assert route.call_count == 2

    http_mod.asyncio.sleep = original_sleep


@respx.mock
async def test_4xx_does_not_retry():
    """404 fails immediately without retrying."""
    envelope = TaskEnvelope(goal="test")
    transport = HTTPTransport(base_url="http://fake-agent", timeout=1.0)

    route = respx.post("http://fake-agent/invoke").mock(
        return_value=httpx.Response(404, text="not found")
    )

    with pytest.raises(RoutingError) as exc_info:
        await transport.invoke(envelope)

    assert route.call_count == 1
    assert "HTTP 404" in str(exc_info.value)
    assert "not retried" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Timeout wired through registry → HTTPAdapter → HTTPTransport
# ---------------------------------------------------------------------------

def test_timeout_saved_in_registry():
    reg.add("agent-a", "http://localhost:9000", ["research"], timeout=120.0)
    entries = reg.list_all()
    assert entries[0]["timeout"] == 120.0


def test_timeout_default_in_registry():
    reg.add("agent-a", "http://localhost:9000", ["research"])
    entries = reg.list_all()
    assert entries[0]["timeout"] == 60.0


def test_timeout_passed_to_http_adapter():
    reg.add("agent-a", "http://localhost:9000", ["research"], timeout=45.0)
    router = reg.load_router()
    # Reach into the adapter to verify timeout was wired through
    adapter = router._agents["agent-a"]
    assert adapter.transport.timeout == 45.0


def test_register_cli_saves_timeout(ping_server):
    result = runner.invoke(app, [
        "register", "agent-a", ping_server,
        "--capabilities", "research",
        "--timeout", "90",
    ])
    assert result.exit_code == 0
    assert "90" in result.output
    entries = reg.list_all()
    assert entries[0]["timeout"] == 90.0


# ---------------------------------------------------------------------------
# oap ping tests
# ---------------------------------------------------------------------------

def test_ping_alive_agent(ping_server):
    runner.invoke(app, ["register", "agent-a", ping_server, "--capabilities", "research"])
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 0
    assert "alive" in result.output
    assert "agent-a" in result.output


def test_ping_dead_agent():
    runner.invoke(app, ["register", "dead-agent", "http://localhost:19990", "--capabilities", "research"])
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 1
    assert "dead" in result.output
    assert "dead-agent" in result.output


def test_ping_mixed_exits_1(ping_server):
    """One alive, one dead → exit code 1."""
    runner.invoke(app, ["register", "alive-agent", ping_server, "--capabilities", "research"])
    runner.invoke(app, ["register", "dead-agent", "http://localhost:19990", "--capabilities", "code"])
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 1
    assert "alive" in result.output
    assert "dead" in result.output


def test_ping_no_health_endpoint(ping_server):
    """A 404 on GET / is 'no health endpoint', not dead — exit 0."""
    # The fake_agent_server returns 404 for GET /
    from tests.fake_agent_server import FakeAgentHandler
    _port = 19995
    server = HTTPServer(("localhost", _port), FakeAgentHandler)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()

    runner.invoke(app, ["register", "no-health", f"http://localhost:{_port}", "--capabilities", "research"])
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 0
    assert "no health endpoint" in result.output

    server.shutdown()


def test_ping_no_agents_registered():
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 0
    assert "No agents registered" in result.output
