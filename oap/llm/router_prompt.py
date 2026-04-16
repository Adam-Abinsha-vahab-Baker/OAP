ROUTING_PROMPT = """You are a routing agent for an AI agent network.
Given a goal and a list of agents with their descriptions,
pick the single best agent to handle this goal.

Goal: {goal}

Available agents:
{agents}

Rules:
- Reply with ONLY the agent_id of the best match
- No explanation, no punctuation, just the agent_id
- If no agent is a good fit, reply: NO_MATCH
"""

def build_prompt(goal: str, agents: list[dict]) -> str:
    agent_lines = "\n".join(
        f"- {a['id']}: {a['description'] or ', '.join(a['capabilities'])}"
        for a in agents
    )
    return ROUTING_PROMPT.format(goal=goal, agents=agent_lines)
