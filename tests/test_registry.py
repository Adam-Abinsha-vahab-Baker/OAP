import pytest
from pathlib import Path
import oap.registry as reg


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY_PATH", tmp_path / "agents.json")


def test_empty_registry_returns_no_entries():
    assert reg.list_all() == []


def test_add_and_list():
    reg.add("agent-a", "http://localhost:9000", ["research", "find"])
    entries = reg.list_all()
    assert len(entries) == 1
    assert entries[0] == {
        "id": "agent-a",
        "url": "http://localhost:9000",
        "capabilities": ["research", "find"],
        "timeout": 60.0,
    }


def test_add_multiple_agents():
    reg.add("agent-a", "http://localhost:9000", ["research"])
    reg.add("agent-b", "http://localhost:9001", ["code"])
    ids = [e["id"] for e in reg.list_all()]
    assert "agent-a" in ids
    assert "agent-b" in ids


def test_add_overwrites_existing():
    reg.add("agent-a", "http://old", ["foo"])
    reg.add("agent-a", "http://new", ["bar"])
    entries = reg.list_all()
    assert len(entries) == 1
    assert entries[0]["url"] == "http://new"
    assert entries[0]["capabilities"] == ["bar"]


def test_remove_existing_agent():
    reg.add("agent-a", "http://localhost:9000", ["research"])
    removed = reg.remove("agent-a")
    assert removed is True
    assert reg.list_all() == []


def test_remove_nonexistent_agent():
    removed = reg.remove("ghost-agent")
    assert removed is False


def test_load_router_returns_registered_agents():
    reg.add("agent-a", "http://localhost:9000", ["research", "find"])
    router = reg.load_router()
    agent_ids = [a["id"] for a in router.list_agents()]
    assert "agent-a" in agent_ids


def test_load_router_empty_registry():
    router = reg.load_router()
    assert router.list_agents() == []


def test_registry_persists_across_calls():
    reg.add("agent-a", "http://localhost:9000", ["research"])
    # simulate a second process reading the same file
    entries = reg.list_all()
    assert len(entries) == 1
