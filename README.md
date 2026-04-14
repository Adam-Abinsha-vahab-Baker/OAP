# OAP — Open Agent Protocol

A lightweight routing layer for passing tasks between AI agents.

OAP defines a standard envelope format (`TaskEnvelope`) and a router that dispatches tasks to the right agent based on capabilities or explicit handoff instructions.

## Installation

```bash
pip install open-agent-protocol
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

# Register an HTTP agent in the local registry
oap register research-agent http://localhost:9000 --capabilities "research,search,find"

# Route the envelope to the best matching agent
oap route task.json --output result.json

# Automatically follow handoffs until the task is complete
oap chain task.json --output final.json

# Inspect an envelope
oap inspect result.json

# Validate envelope structure
oap validate result.json

# List all registered agents
oap agents

# Remove an agent from the registry
oap unregister research-agent
```

## Chaining agents

When an agent sets a `handoff.next_agent` on its response, `oap chain` automatically routes to the next agent and keeps going until the task is complete or a hop limit is reached.

```python
# Python API
result, visited = await router.chain(envelope, max_hops=10)
print(" → ".join(visited))  # e.g. research-agent → summarise-agent
```

```bash
# CLI — follows handoffs automatically, prints each hop
oap chain task.json --output final.json --max-hops 5
```

The chain stops when:
- An agent returns a response with no `handoff` set, or
- `max_hops` is reached (default: 10)

## Registry

Agents are stored in `~/.oap/agents.json` and persist across commands.

```bash
oap register my-agent http://localhost:9000 --capabilities "research,find"
oap agents       # list all registered agents
oap unregister my-agent
```

## Concepts

- **TaskEnvelope** — the standard task object passed between agents. Contains the goal, memory, steps taken, and optional constraints.
- **OAPRouter** — selects the best registered agent for a given envelope and invokes it.
- **AgentAdapter** — translates between the envelope format and an agent's native interface.
- **HTTPAdapter** — built-in adapter for agents that expose a `POST /invoke` endpoint.

## License

MIT
