"""Tests for the self-describing oap register command."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from typer.testing import CliRunner

from oap.cli import app
import oap.registry as reg

runner = CliRunner()

_REG_PORT = 19993
_REG_PORT_NO_HEALTH = 19992
_REG_PORT_NO_CAPS = 19991


# ---------------------------------------------------------------------------
# Fake servers
# ---------------------------------------------------------------------------

class SelfDescribingHandler(BaseHTTPRequestHandler):
    """Returns full agent info on GET /, accepts POST /invoke."""

    agent_info = {
        "agent_id": "research-agent",
        "capabilities": ["research", "find", "search", "analyse"],
        "description": "Researches topics thoroughly.",
        "status": "ok",
    }

    def do_GET(self):
        body = json.dumps(self.agent_info).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        body = json.dumps({"result": "done", "memory": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class NoCapsHandler(BaseHTTPRequestHandler):
    """GET / returns 200 but no capabilities field."""

    def do_GET(self):
        body = json.dumps({"agent_id": "empty-agent", "status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def self_describing_server():
    server = HTTPServer(("localhost", _REG_PORT), SelfDescribingHandler)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()
    yield f"http://localhost:{_REG_PORT}"
    server.shutdown()


@pytest.fixture(scope="module")
def no_caps_server():
    server = HTTPServer(("localhost", _REG_PORT_NO_CAPS), NoCapsHandler)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()
    yield f"http://localhost:{_REG_PORT_NO_CAPS}"
    server.shutdown()


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY_PATH", tmp_path / "agents.json")


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------

def test_register_discovers_capabilities(self_describing_server):
    result = runner.invoke(app, ["register", "my-agent", self_describing_server])
    assert result.exit_code == 0, result.output
    assert "Discovered" in result.output

    entries = reg.list_all()
    assert len(entries) == 1
    assert sorted(entries[0]["capabilities"]) == sorted(["research", "find", "search", "analyse"])


def test_register_discovers_description(self_describing_server):
    runner.invoke(app, ["register", "my-agent", self_describing_server])
    entries = reg.list_all()
    assert entries[0]["description"] == "Researches topics thoroughly."


def test_register_uses_agent_id_from_response(self_describing_server):
    """The agent_id from GET / should override the CLI positional argument."""
    runner.invoke(app, ["register", "cli-given-id", self_describing_server])
    entries = reg.list_all()
    assert entries[0]["id"] == "research-agent"


def test_register_fallback_to_capabilities_flag_on_no_endpoint():
    """GET / fails (nothing listening) → fall back to --capabilities."""
    result = runner.invoke(app, [
        "register", "my-agent", "http://localhost:19980",
        "--capabilities", "research,find",
    ])
    assert result.exit_code == 0, result.output
    entries = reg.list_all()
    assert sorted(entries[0]["capabilities"]) == sorted(["research", "find"])


def test_register_error_when_no_endpoint_and_no_caps():
    """GET / fails and no --capabilities → clear error, exit 1."""
    result = runner.invoke(app, ["register", "my-agent", "http://localhost:19980"])
    assert result.exit_code == 1
    assert "no GET / endpoint" in result.output
    assert "--capabilities" in result.output


def test_register_fallback_when_no_caps_in_response(no_caps_server):
    """GET / returns 200 but no capabilities → fall back to --capabilities."""
    result = runner.invoke(app, [
        "register", "my-agent", no_caps_server,
        "--capabilities", "research,find",
    ])
    assert result.exit_code == 0, result.output
    entries = reg.list_all()
    assert "research" in entries[0]["capabilities"]


def test_register_error_when_no_caps_in_response_and_no_flag(no_caps_server):
    """GET / returns 200 but no capabilities and no --capabilities → exit 1."""
    result = runner.invoke(app, ["register", "my-agent", no_caps_server])
    assert result.exit_code == 1
    assert "returned no capabilities" in result.output


def test_register_saves_description_as_empty_for_manual_registration():
    """Manual --capabilities registration stores empty description."""
    runner.invoke(app, [
        "register", "my-agent", "http://localhost:19980",
        "--capabilities", "research",
    ])
    entries = reg.list_all()
    assert entries[0]["description"] == ""


# ---------------------------------------------------------------------------
# ping capability update tests
# ---------------------------------------------------------------------------

def test_ping_updates_capabilities_when_changed(self_describing_server):
    """ping re-reads GET / and updates capabilities when they differ."""
    # Register with stale capabilities
    reg.add("research-agent", self_describing_server, ["old-cap"], description="old")

    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 0, result.output
    assert "capabilities updated" in result.output

    entries = reg.list_all()
    assert sorted(entries[0]["capabilities"]) == sorted(["research", "find", "search", "analyse"])


def test_ping_no_update_when_capabilities_match(self_describing_server):
    """ping does not show 'capabilities updated' when nothing changed."""
    reg.add(
        "research-agent", self_describing_server,
        ["research", "find", "search", "analyse"],
        description="Researches topics thoroughly.",
    )
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 0, result.output
    assert "capabilities updated" not in result.output


# ---------------------------------------------------------------------------
# oap agents description column
# ---------------------------------------------------------------------------

def test_agents_shows_description_column(self_describing_server):
    runner.invoke(app, ["register", "my-agent", self_describing_server])
    result = runner.invoke(app, ["agents"])
    assert result.exit_code == 0
    assert "Description" in result.output
    assert "Researches" in result.output


def test_agents_shows_dash_for_empty_description():
    reg.add("my-agent", "http://localhost:9000", ["research"], description="")
    result = runner.invoke(app, ["agents"])
    assert result.exit_code == 0
    assert "—" in result.output


# ---------------------------------------------------------------------------
# Backward compat: old registry entries without description
# ---------------------------------------------------------------------------

def test_list_all_defaults_description_for_old_entries(tmp_path, monkeypatch):
    """Registry entries written before description field was added default to ''."""
    registry_file = tmp_path / "agents.json"
    registry_file.write_text(json.dumps({
        "legacy-agent": {"url": "http://localhost:9000", "capabilities": ["research"], "timeout": 60.0}
    }))
    monkeypatch.setattr(reg, "_REGISTRY_PATH", registry_file)

    entries = reg.list_all()
    assert entries[0]["description"] == ""
