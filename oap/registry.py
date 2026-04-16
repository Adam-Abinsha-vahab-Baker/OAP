from __future__ import annotations

import json
from pathlib import Path

from oap.adapters.http import HTTPAdapter
from oap.router import OAPRouter
from oap.llm.factory import get_provider

_REGISTRY_PATH = Path.home() / ".oap" / "agents.json"
_DEFAULT_TIMEOUT = 60.0


def _load_raw() -> dict[str, dict]:
    if not _REGISTRY_PATH.exists():
        return {}
    return json.loads(_REGISTRY_PATH.read_text())


def _save_raw(data: dict[str, dict]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2))


def add(
    agent_id: str,
    url: str,
    capabilities: list[str],
    timeout: float = _DEFAULT_TIMEOUT,
    description: str = "",
) -> None:
    data = _load_raw()
    data[agent_id] = {
        "url": url,
        "capabilities": capabilities,
        "timeout": timeout,
        "description": description,
    }
    _save_raw(data)


def remove(agent_id: str) -> bool:
    data = _load_raw()
    if agent_id not in data:
        return False
    del data[agent_id]
    _save_raw(data)
    return True


def list_all() -> list[dict]:
    return [
        {
            "id": agent_id,
            "url": entry["url"],
            "capabilities": entry["capabilities"],
            "timeout": entry.get("timeout", _DEFAULT_TIMEOUT),
            "description": entry.get("description", ""),
        }
        for agent_id, entry in _load_raw().items()
    ]


def load_router() -> OAPRouter:
    provider = get_provider()
    router = OAPRouter(llm_provider=provider)
    for agent_id, entry in _load_raw().items():
        timeout = entry.get("timeout", _DEFAULT_TIMEOUT)
        router.register(
            agent_id,
            HTTPAdapter(agent_id=agent_id, base_url=entry["url"], timeout=timeout),
            entry["capabilities"],
            description=entry.get("description", ""),
        )
    return router
