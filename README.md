# OAP — Open Agent Protocol

A lightweight routing layer for passing tasks between AI agents.

OAP defines a standard envelope format (`TaskEnvelope`) and a router that dispatches tasks to the right agent based on capabilities or explicit handoff instructions.

## Installation

```bash
pip install oap
```

## Quick start

```python
from oap import TaskEnvelope, OAPRouter
from oap.adapters.http import HTTPAdapter

router = OAPRouter()
router.register(
    "research-agent",
    HTTPAdapter(agent_id="research-agent", base_url="http://localhost:9000"),
    capabilities=["research", "search", "find"],
)

envelope = TaskEnvelope(goal="research the best vector databases")
result = await router.route(envelope)
print(result.memory["last_result"])
```

## CLI

```bash
# Create a new task envelope
oap init "research the best vector databases" --output task.json

# Route it to an HTTP agent
oap register research-agent http://localhost:9000 --capabilities "research,search" task.json --output result.json

# Inspect the result
oap inspect result.json

# Validate envelope structure
oap validate result.json

# Route using built-in demo agents
oap route task.json

# List demo agents
oap agents
```

## Concepts

- **TaskEnvelope** — the standard task object passed between agents. Contains the goal, memory, steps taken, and optional constraints.
- **OAPRouter** — selects the best registered agent for a given envelope and invokes it.
- **AgentAdapter** — translates between the envelope format and an agent's native interface.
- **HTTPAdapter** — built-in adapter for agents that expose a `POST /invoke` endpoint.

## License

MIT
