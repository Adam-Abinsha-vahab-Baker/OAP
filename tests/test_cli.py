import json
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fake_agent_server import FakeAgentHandler, HTTPServer
from oap.cli import app
import oap.registry as reg

runner = CliRunner()

_CLI_TEST_PORT = 19998


@pytest.fixture(scope="module")
def fake_server():
    server = HTTPServer(("localhost", _CLI_TEST_PORT), FakeAgentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://localhost:{_CLI_TEST_PORT}"
    server.shutdown()


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Point the registry at a temp file for every test."""
    registry_file = tmp_path / "agents.json"
    monkeypatch.setattr(reg, "_REGISTRY_PATH", registry_file)


# --- init ---

def test_init_prints_envelope():
    result = runner.invoke(app, ["init", "test goal"])
    assert result.exit_code == 0
    assert "test goal" in result.output


def test_init_saves_to_file(tmp_path):
    out = tmp_path / "envelope.json"
    result = runner.invoke(app, ["init", "save this goal", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["goal"] == "save this goal"


# --- inspect ---

def test_inspect_valid_file(tmp_path):
    out = tmp_path / "envelope.json"
    runner.invoke(app, ["init", "inspect me", "--output", str(out)])
    result = runner.invoke(app, ["inspect", str(out)])
    assert result.exit_code == 0
    assert "inspect me" in result.output


def test_inspect_missing_file():
    result = runner.invoke(app, ["inspect", "/nonexistent/path.json"])
    assert result.exit_code == 1


# --- validate ---

def test_validate_valid_file(tmp_path):
    out = tmp_path / "envelope.json"
    runner.invoke(app, ["init", "valid goal", "--output", str(out)])
    result = runner.invoke(app, ["validate", str(out)])
    assert result.exit_code == 0
    assert "Valid" in result.output


def test_validate_invalid_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "an envelope"}')
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1


# --- register / unregister / agents ---

def test_register_saves_to_registry():
    result = runner.invoke(app, [
        "register", "research-agent", "http://localhost:9000",
        "--capabilities", "research,find,search",
    ])
    assert result.exit_code == 0
    assert "research-agent" in result.output
    entries = reg.list_all()
    assert len(entries) == 1
    assert entries[0]["id"] == "research-agent"
    assert entries[0]["url"] == "http://localhost:9000"
    assert "research" in entries[0]["capabilities"]


def test_register_overwrites_existing():
    runner.invoke(app, ["register", "agent-x", "http://old", "--capabilities", "foo"])
    runner.invoke(app, ["register", "agent-x", "http://new", "--capabilities", "bar"])
    entries = reg.list_all()
    assert len(entries) == 1
    assert entries[0]["url"] == "http://new"


def test_unregister_removes_agent():
    runner.invoke(app, ["register", "agent-y", "http://localhost:1234", "--capabilities", "foo"])
    result = runner.invoke(app, ["unregister", "agent-y"])
    assert result.exit_code == 0
    assert reg.list_all() == []


def test_unregister_missing_agent():
    result = runner.invoke(app, ["unregister", "nonexistent"])
    assert result.exit_code == 1


def test_agents_empty_registry():
    result = runner.invoke(app, ["agents"])
    assert result.exit_code == 0
    assert "No agents registered" in result.output


def test_agents_lists_registered():
    runner.invoke(app, ["register", "research-agent", "http://localhost:9000", "--capabilities", "research,find"])
    runner.invoke(app, ["register", "coding-agent", "http://localhost:9001", "--capabilities", "code,debug"])
    result = runner.invoke(app, ["agents"])
    assert result.exit_code == 0
    assert "research-agent" in result.output
    assert "coding-agent" in result.output


# --- route ---

def test_route_dry_run(tmp_path, fake_server):
    task = tmp_path / "task.json"
    runner.invoke(app, ["init", "research neural networks", "--output", str(task)])
    runner.invoke(app, ["register", "research-agent", fake_server, "--capabilities", "research,find,search"])
    result = runner.invoke(app, ["route", str(task), "--dry-run"])
    assert result.exit_code == 0
    assert "research-agent" in result.output


def test_route_produces_result(tmp_path, fake_server):
    task = tmp_path / "task.json"
    result_file = tmp_path / "result.json"
    runner.invoke(app, ["init", "research the best vector databases", "--output", str(task)])
    runner.invoke(app, ["register", "research-agent", fake_server, "--capabilities", "research,find,search"])
    result = runner.invoke(app, ["route", str(task), "--output", str(result_file)])
    assert result.exit_code == 0, result.output
    assert result_file.exists()
    data = json.loads(result_file.read_text())
    assert len(data["steps_taken"]) == 1
    assert data["steps_taken"][0]["agent_id"] == "research-agent"


def test_route_no_agents_registered(tmp_path):
    task = tmp_path / "task.json"
    runner.invoke(app, ["init", "research something", "--output", str(task)])
    result = runner.invoke(app, ["route", str(task)])
    assert result.exit_code == 1
    assert "Routing failed" in result.output


def test_route_missing_file():
    result = runner.invoke(app, ["route", "/nonexistent/task.json"])
    assert result.exit_code == 1


def test_route_invalid_envelope(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "valid"}')
    result = runner.invoke(app, ["route", str(bad)])
    assert result.exit_code == 1
