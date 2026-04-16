# OAP — Open Agent Protocol

A lightweight routing layer for passing tasks between AI agents.

OAP defines a standard envelope format (`TaskEnvelope`) and a router that dispatches tasks to the right agent based on capabilities or explicit handoff instructions. Routing can use keyword matching (default) or an LLM provider for semantic matching.

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

# Register an HTTP agent — OAP discovers capabilities automatically from GET /
oap register research-agent http://localhost:9000
# Falls back to manual capabilities if the agent has no GET / endpoint
oap register research-agent http://localhost:9000 --capabilities "research,search,find"

# Route the envelope to the best matching agent
oap route task.json --output result.json

# Automatically follow handoffs until the task is complete
oap chain task.json --output final.json

# List all interrupted chains (saved mid-flight)
oap runs

# Resume a chain that failed or was interrupted
oap resume <envelope-id>
oap resume <envelope-id> --pipeline "agent-a,agent-b,agent-c"

# Inspect an envelope
oap inspect result.json

# Validate envelope structure
oap validate result.json

# List all registered agents
oap agents

# Check reachability of all registered agents
oap ping

# Remove an agent from the registry
oap unregister research-agent
```

## LLM routing

By default OAP routes using keyword matching against agent capabilities. You can upgrade to semantic LLM-based routing by configuring a provider — OAP will use the agent's `description` field to pick the best match.

```bash
# Configure a provider
oap config set-llm openai --model gpt-4o-mini   # OpenAI
oap config set-llm bedrock                        # AWS Bedrock (uses machine credentials)
oap config set-llm ollama --model llama3          # local Ollama
oap config set-llm custom                         # any OpenAI-compatible endpoint

# Verify it works
oap config test-llm

# Inspect current config
oap config show

# Remove LLM config — falls back to keyword matching
oap config clear-llm
```

### Supported providers

| Provider | Auth | Key env var |
|----------|------|-------------|
| `openai` | API key | `OPENAI_API_KEY` or `OAP_OPENAI_API_KEY` |
| `bedrock` | AWS credentials | standard AWS credential chain |
| `ollama` | none (local) | `OAP_OLLAMA_URL` (optional, default `http://localhost:11434`) |
| `custom` | optional key | `OAP_CUSTOM_BASE_URL`, `OAP_CUSTOM_API_KEY`, `OAP_CUSTOM_MODEL` |

Keys are **never stored on disk** — they are read from environment variables at runtime. Only the provider name and model are saved to `~/.oap/config.json`.

### Python SDK

```python
from oap import OAPRouter
from oap.llm import get_provider

# Auto-detects from ~/.oap/config.json
router = OAPRouter(llm_provider=get_provider())

# Or explicit
from oap.llm.openai import OpenAIProvider
router = OAPRouter(llm_provider=OpenAIProvider(model="gpt-4o-mini"))

from oap.llm.bedrock import BedrockProvider
router = OAPRouter(llm_provider=BedrockProvider())

from oap.llm.ollama import OllamaProvider
router = OAPRouter(llm_provider=OllamaProvider(model="llama3"))

# Keyword matching only — no provider
router = OAPRouter()
```

When an LLM provider is configured, OAP tries LLM routing first. If the LLM returns an unknown agent or fails, it automatically falls back to keyword matching. Explicit `handoff.next_agent` always takes priority over both.

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

## Resuming failed chains

If a chain is interrupted (network error, agent down, hop limit hit), OAP saves progress after each hop. Resume from the last successful hop with:

```bash
oap runs                        # list all interrupted chains
oap resume <envelope-id>        # resume handoff-driven chain
oap resume <envelope-id> --pipeline "agent-a,agent-b,agent-c"  # resume a fixed pipeline
oap resume <envelope-id> --max-hops 5  # cap additional hops
```

Progress is stored in `~/.oap/runs/` and deleted automatically when the chain completes.

## Registry

Agents are stored in `~/.oap/agents.json` and persist across commands.

When an agent implements `GET /` returning `{agent_id, capabilities, description}`, registration is automatic:

```bash
oap register my-agent http://localhost:9000
# → OAP hits GET /, reads capabilities and description, saves everything

oap register my-agent http://localhost:9000 --capabilities "research,find"
# → fallback: use provided capabilities if GET / is unavailable

oap agents          # list all registered agents with description column
oap ping            # health-check all agents, auto-updates capabilities if changed
oap unregister my-agent
```

### Agent health endpoint

Add `GET /` to your agent to enable self-registration and health checks:

```python
@app.get("/")
async def info():
    return {
        "agent_id": "my-agent",
        "capabilities": ["research", "find", "search"],
        "description": "Researches topics and returns structured findings.",
        "status": "ok",
    }
```

The `description` field is used by LLM routing to semantically match agents to goals.

## Concepts

- **TaskEnvelope** — the standard task object passed between agents. Contains the goal, memory, steps taken, and optional constraints.
- **OAPRouter** — selects the best registered agent for a given envelope and invokes it.
- **AgentAdapter** — translates between the envelope format and an agent's native interface.
- **HTTPAdapter** — built-in adapter for agents that expose a `POST /invoke` endpoint.
- **LLMProvider** — optional routing backend. Implement `complete(prompt)` and `is_available()` to add a custom provider.

## License

MIT
