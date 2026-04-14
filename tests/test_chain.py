import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from oap.cli import app
from oap.envelope import TaskEnvelope
from oap.router import OAPRouter, RoutingError
from oap.adapters.mock import MockAgentAdapter
import oap.registry as reg

runner = CliRunner()

_CHAIN_PORT = 19997


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_router(*agents) -> OAPRouter:
    """Build a router from (agent_id, MockAgentAdapter, capabilities) tuples."""
    router = OAPRouter()
    for agent_id, adapter, caps in agents:
        router.register(agent_id, adapter, caps)
    return router


# ---------------------------------------------------------------------------
# Router-level chain() tests
# ---------------------------------------------------------------------------

async def test_chain_single_hop_no_handoff():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", response="done"), ["research"]),
    )
    envelope = TaskEnvelope(goal="research something")
    result, visited = await router.chain(envelope)

    assert visited == ["agent-a"]
    assert result.handoff is None
    assert len(result.steps_taken) == 1


async def test_chain_follows_handoff():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", response="step 1", next_agent="agent-b"), ["research"]),
        ("agent-b", MockAgentAdapter("agent-b", response="step 2"), ["code"]),
    )
    envelope = TaskEnvelope(goal="research neural networks")
    result, visited = await router.chain(envelope)

    assert visited == ["agent-a", "agent-b"]
    assert result.handoff is None
    assert len(result.steps_taken) == 2
    assert result.memory["last_result"] == "step 2"


async def test_chain_three_hops():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", next_agent="agent-b"), ["find"]),
        ("agent-b", MockAgentAdapter("agent-b", next_agent="agent-c"), ["research"]),
        ("agent-c", MockAgentAdapter("agent-c", response="final"), ["summarise"]),
    )
    envelope = TaskEnvelope(goal="find relevant papers")
    result, visited = await router.chain(envelope)

    assert visited == ["agent-a", "agent-b", "agent-c"]
    assert result.memory["last_result"] == "final"


async def test_chain_stops_at_max_hops():
    # agent-a always hands off to itself
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", next_agent="agent-a"), ["research"]),
    )
    envelope = TaskEnvelope(goal="research forever")
    result, visited = await router.chain(envelope, max_hops=3)

    assert len(visited) == 3


async def test_chain_on_hop_callback():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", next_agent="agent-b"), ["research"]),
        ("agent-b", MockAgentAdapter("agent-b"), ["code"]),
    )
    envelope = TaskEnvelope(goal="research neural networks")
    hops_seen = []
    result, visited = await router.chain(
        envelope, on_hop=lambda hop, aid: hops_seen.append((hop, aid))
    )

    assert hops_seen == [(1, "agent-a"), (2, "agent-b")]


async def test_chain_raises_routing_error_when_no_agents():
    router = OAPRouter()
    envelope = TaskEnvelope(goal="anything")
    with pytest.raises(RoutingError):
        await router.chain(envelope)


async def test_chain_raises_routing_error_on_unknown_handoff():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", next_agent="ghost"), ["research"]),
    )
    envelope = TaskEnvelope(goal="research something")
    with pytest.raises(RoutingError):
        await router.chain(envelope)


async def test_chain_preserves_goal_and_id():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", next_agent="agent-b"), ["research"]),
        ("agent-b", MockAgentAdapter("agent-b"), ["code"]),
    )
    envelope = TaskEnvelope(goal="research neural networks")
    result, _ = await router.chain(envelope)

    assert result.goal == envelope.goal
    assert result.id == envelope.id


# ---------------------------------------------------------------------------
# Fake HTTP server that supports handoff responses
# ---------------------------------------------------------------------------

class HandoffAgentHandler(BaseHTTPRequestHandler):
    """On first call: returns a handoff to 'second-agent'. On second call: returns done."""
    call_count = 0

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)

        HandoffAgentHandler.call_count += 1
        if HandoffAgentHandler.call_count == 1:
            body = json.dumps({
                "result": "first hop done",
                "handoff": {"next_agent": "second-agent", "reason": "needs more work"},
            })
        else:
            body = json.dumps({"result": "second hop done"})

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def handoff_server():
    HandoffAgentHandler.call_count = 0
    server = HTTPServer(("localhost", _CHAIN_PORT), HandoffAgentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://localhost:{_CHAIN_PORT}"
    server.shutdown()


# ---------------------------------------------------------------------------
# CLI chain command tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY_PATH", tmp_path / "agents.json")


def test_chain_cli_single_hop(tmp_path, handoff_server):
    HandoffAgentHandler.call_count = 1  # skip handoff branch

    task = tmp_path / "task.json"
    out = tmp_path / "final.json"
    runner.invoke(app, ["init", "research something", "--output", str(task)])
    runner.invoke(app, ["register", "second-agent", handoff_server, "--capabilities", "research"])

    result = runner.invoke(app, ["chain", str(task), "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Done" in result.output


def test_chain_cli_follows_handoff(tmp_path, handoff_server):
    HandoffAgentHandler.call_count = 0  # start from handoff branch

    task = tmp_path / "task.json"
    out = tmp_path / "final.json"
    runner.invoke(app, ["init", "research something", "--output", str(task)])
    # Register both agents pointing at the same fake server
    runner.invoke(app, ["register", "first-agent", handoff_server, "--capabilities", "research"])
    runner.invoke(app, ["register", "second-agent", handoff_server, "--capabilities", "second"])

    result = runner.invoke(app, ["chain", str(task), "--output", str(out)])

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert len(data["steps_taken"]) == 2


def test_chain_cli_missing_file():
    result = runner.invoke(app, ["chain", "/nonexistent/task.json"])
    assert result.exit_code == 1


def test_chain_cli_invalid_envelope(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "valid"}')
    result = runner.invoke(app, ["chain", str(bad)])
    assert result.exit_code == 1


def test_chain_cli_no_agents_registered(tmp_path):
    task = tmp_path / "task.json"
    runner.invoke(app, ["init", "research something", "--output", str(task)])
    result = runner.invoke(app, ["chain", str(task)])
    assert result.exit_code == 1
    assert "Routing failed" in result.output


def test_chain_cli_max_hops_flag(tmp_path, handoff_server):
    HandoffAgentHandler.call_count = 0

    task = tmp_path / "task.json"
    runner.invoke(app, ["init", "research something", "--output", str(task)])
    runner.invoke(app, ["register", "first-agent", handoff_server, "--capabilities", "research"])
    runner.invoke(app, ["register", "second-agent", handoff_server, "--capabilities", "second"])

    result = runner.invoke(app, ["chain", str(task), "--max-hops", "1"])

    assert result.exit_code == 0, result.output
    assert "1 hop" in result.output


# ---------------------------------------------------------------------------
# Router-level run_pipeline() tests
# ---------------------------------------------------------------------------

async def test_pipeline_single_agent():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", response="done"), ["research"]),
    )
    envelope = TaskEnvelope(goal="research something")
    result, visited = await router.run_pipeline(envelope, ["agent-a"])

    assert visited == ["agent-a"]
    assert len(result.steps_taken) == 1
    assert result.memory["last_result"] == "done"


async def test_pipeline_three_hops_in_order():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", response="step 1"), ["research"]),
        ("agent-b", MockAgentAdapter("agent-b", response="step 2"), ["write"]),
        ("agent-c", MockAgentAdapter("agent-c", response="step 3"), ["translate"]),
    )
    envelope = TaskEnvelope(goal="do a thing")
    result, visited = await router.run_pipeline(envelope, ["agent-a", "agent-b", "agent-c"])

    assert visited == ["agent-a", "agent-b", "agent-c"]
    assert len(result.steps_taken) == 3
    assert result.memory["last_result"] == "step 3"


async def test_pipeline_ignores_handoff():
    # agent-a wants to hand off to agent-c, but pipeline forces agent-b next
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a", response="step 1", next_agent="agent-c"), ["research"]),
        ("agent-b", MockAgentAdapter("agent-b", response="step 2"), ["write"]),
    )
    envelope = TaskEnvelope(goal="research something")
    result, visited = await router.run_pipeline(envelope, ["agent-a", "agent-b"])

    assert visited == ["agent-a", "agent-b"]
    assert result.handoff is None


async def test_pipeline_unknown_agent_raises():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a"), ["research"]),
    )
    envelope = TaskEnvelope(goal="do something")
    with pytest.raises(RoutingError, match="not found in registry"):
        await router.run_pipeline(envelope, ["agent-a", "ghost-agent"])


async def test_pipeline_preserves_goal_and_id():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a"), ["research"]),
        ("agent-b", MockAgentAdapter("agent-b"), ["write"]),
    )
    envelope = TaskEnvelope(goal="research something")
    result, _ = await router.run_pipeline(envelope, ["agent-a", "agent-b"])

    assert result.goal == envelope.goal
    assert result.id == envelope.id


async def test_pipeline_step_labels():
    router = make_router(
        ("agent-a", MockAgentAdapter("agent-a"), ["research"]),
        ("agent-b", MockAgentAdapter("agent-b"), ["write"]),
    )
    envelope = TaskEnvelope(goal="research something")
    result, _ = await router.run_pipeline(envelope, ["agent-a", "agent-b"])

    assert "1/2" in result.steps_taken[0].action
    assert "2/2" in result.steps_taken[1].action


# ---------------------------------------------------------------------------
# CLI --pipeline tests
# ---------------------------------------------------------------------------

def test_pipeline_cli_three_hops(tmp_path, handoff_server):
    HandoffAgentHandler.call_count = 99  # force "done" response for all calls

    task = tmp_path / "task.json"
    out = tmp_path / "final.json"
    runner.invoke(app, ["init", "do a thing", "--output", str(task)])
    runner.invoke(app, ["register", "agent-a", handoff_server, "--capabilities", "research"])
    runner.invoke(app, ["register", "agent-b", handoff_server, "--capabilities", "write"])
    runner.invoke(app, ["register", "agent-c", handoff_server, "--capabilities", "translate"])

    result = runner.invoke(app, [
        "chain", str(task),
        "--pipeline", "agent-a,agent-b,agent-c",
        "--output", str(out),
    ])

    assert result.exit_code == 0, result.output
    assert "hop 1/3" in result.output
    assert "hop 2/3" in result.output
    assert "hop 3/3" in result.output
    assert "3 hop(s)" in result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data["steps_taken"]) == 3


def test_pipeline_cli_single_agent(tmp_path, handoff_server):
    HandoffAgentHandler.call_count = 99

    task = tmp_path / "task.json"
    out = tmp_path / "final.json"
    runner.invoke(app, ["init", "do a thing", "--output", str(task)])
    runner.invoke(app, ["register", "agent-a", handoff_server, "--capabilities", "research"])

    result = runner.invoke(app, [
        "chain", str(task),
        "--pipeline", "agent-a",
        "--output", str(out),
    ])

    assert result.exit_code == 0, result.output
    assert "1 hop(s)" in result.output
    assert out.exists()


def test_pipeline_cli_unknown_agent(tmp_path, handoff_server):
    task = tmp_path / "task.json"
    runner.invoke(app, ["init", "do a thing", "--output", str(task)])
    runner.invoke(app, ["register", "agent-a", handoff_server, "--capabilities", "research"])

    result = runner.invoke(app, [
        "chain", str(task),
        "--pipeline", "agent-a,ghost-agent",
    ])

    assert result.exit_code == 1
    assert "Routing failed" in result.output
    assert "ghost-agent" in result.output


def test_pipeline_cli_output_saved(tmp_path, handoff_server):
    HandoffAgentHandler.call_count = 99

    task = tmp_path / "task.json"
    out = tmp_path / "final.json"
    runner.invoke(app, ["init", "do a thing", "--output", str(task)])
    runner.invoke(app, ["register", "agent-a", handoff_server, "--capabilities", "research"])

    runner.invoke(app, [
        "chain", str(task),
        "--pipeline", "agent-a",
        "--output", str(out),
    ])

    assert out.exists()
    data = json.loads(out.read_text())
    assert data["goal"] == "do a thing"
    assert len(data["steps_taken"]) == 1
