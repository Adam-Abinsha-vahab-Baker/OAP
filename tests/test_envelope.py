import pytest
from oap.envelope import TaskEnvelope, Step, Constraints, Handoff


def test_default_envelope():
    e = TaskEnvelope(goal="test goal")
    assert e.goal == "test goal"
    assert e.version == "0.1"
    assert e.memory == {}
    assert e.steps_taken == []
    assert e.handoff is None


def test_envelope_id_is_unique():
    a = TaskEnvelope(goal="task a")
    b = TaskEnvelope(goal="task b")
    assert a.id != b.id


def test_add_step():
    e = TaskEnvelope(goal="test")
    e.add_step(agent_id="research-agent", action="searched web", result="5 results")
    assert len(e.steps_taken) == 1
    assert e.steps_taken[0].agent_id == "research-agent"
    assert e.steps_taken[0].result == "5 results"


def test_add_multiple_steps():
    e = TaskEnvelope(goal="test")
    e.add_step("agent-a", "step 1")
    e.add_step("agent-b", "step 2")
    assert len(e.steps_taken) == 2


def test_with_handoff():
    e = TaskEnvelope(goal="test")
    e.with_handoff(next_agent="coding-agent", reason="needs code", partial_result={"found": True})
    assert e.handoff is not None
    assert e.handoff.next_agent == "coding-agent"
    assert e.handoff.reason == "needs code"
    assert e.handoff.partial_result == {"found": True}


def test_serialise_roundtrip():
    e = TaskEnvelope(goal="roundtrip test")
    e.add_step("agent-x", "did something", result="ok")
    json_str = e.model_dump_json()
    restored = TaskEnvelope.model_validate_json(json_str)
    assert restored.id == e.id
    assert restored.goal == e.goal
    assert len(restored.steps_taken) == 1


def test_constraints_defaults():
    e = TaskEnvelope(goal="test")
    assert e.constraints.max_cost_usd is None
    assert e.constraints.allowed_tools is None
    assert e.constraints.deadline_ms is None


def test_invalid_envelope_missing_goal():
    with pytest.raises(Exception):
        TaskEnvelope.model_validate({"version": "0.1"})