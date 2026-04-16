from __future__ import annotations
import json
from pathlib import Path

CONFIG_FILE = Path.home() / ".oap" / "config.json"


def load() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def save(data: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def get_llm_config() -> dict | None:
    return load().get("llm")


def set_llm_config(provider: str, model: str | None = None) -> None:
    data = load()
    data["llm"] = {"provider": provider}
    if model:
        data["llm"]["model"] = model
    save(data)


def clear_llm_config() -> None:
    data = load()
    data.pop("llm", None)
    save(data)
