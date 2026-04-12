import json
import threading
import pytest
from typer.testing import CliRunner
from tests.fake_agent_server import FakeAgentHandler, HTTPServer
from oap.cli import app

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


def test_inspect_valid_file(tmp_path):
    out = tmp_path / "envelope.json"
    runner.invoke(app, ["init", "inspect me", "--output", str(out)])
    result = runner.invoke(app, ["inspect", str(out)])
    assert result.exit_code == 0
    assert "inspect me" in result.output


def test_inspect_missing_file():
    result = runner.invoke(app, ["inspect", "/nonexistent/path.json"])
    assert result.exit_code == 1


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


def test_route_dry_run(tmp_path):
    out = tmp_path / "task.json"
    runner.invoke(app, ["init", "research neural networks", "--output", str(out)])
    result = runner.invoke(app, ["route", str(out), "--dry-run"])
    assert result.exit_code == 0
    assert "research-agent" in result.output


def test_route_produces_result(tmp_path):
    task = tmp_path / "task.json"
    result_file = tmp_path / "result.json"
    runner.invoke(app, ["init", "debug my python code", "--output", str(task)])
    result = runner.invoke(app, ["route", str(task), "--output", str(result_file)])
    assert result.exit_code == 0
    assert result_file.exists()
    data = json.loads(result_file.read_text())
    assert len(data["steps_taken"]) == 1
    assert data["steps_taken"][0]["agent_id"] == "coding-agent"


def test_agents_lists_registered():
    result = runner.invoke(app, ["agents"])
    assert result.exit_code == 0
    assert "research-agent" in result.output
    assert "coding-agent" in result.output


# --- register command ---

def test_register_routes_to_http_agent(tmp_path, fake_server):
    task = tmp_path / "task.json"
    result_file = tmp_path / "result.json"
    runner.invoke(app, ["init", "research the best vector databases", "--output", str(task)])

    result = runner.invoke(app, [
        "register", "research-agent", fake_server,
        "--capabilities", "research,find,search",
        str(task),
        "--output", str(result_file),
    ])

    assert result.exit_code == 0, result.output
    assert result_file.exists()
    data = json.loads(result_file.read_text())
    assert len(data["steps_taken"]) == 1
    assert data["steps_taken"][0]["agent_id"] == "research-agent"


def test_register_missing_file(fake_server):
    result = runner.invoke(app, [
        "register", "research-agent", fake_server,
        "--capabilities", "research",
        "/nonexistent/task.json",
    ])
    assert result.exit_code == 1


def test_register_invalid_envelope(tmp_path, fake_server):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "valid"}')
    result = runner.invoke(app, [
        "register", "research-agent", fake_server,
        "--capabilities", "research",
        str(bad),
    ])
    assert result.exit_code == 1