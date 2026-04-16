from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from oap.envelope import TaskEnvelope

RUNS_DIR = Path.home() / ".oap" / "runs"


def _path(envelope_id: str) -> Path:
    return RUNS_DIR / f"{envelope_id}.json"


def save(envelope: TaskEnvelope) -> None:
    """Persist the current envelope state after a successful hop."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _path(envelope.id).write_text(envelope.model_dump_json(indent=2))


def load(envelope_id: str) -> TaskEnvelope | None:
    """Load a saved run. Returns None if not found."""
    p = _path(envelope_id)
    if not p.exists():
        return None
    return TaskEnvelope.model_validate_json(p.read_text())


def list_runs() -> list[dict]:
    """Return all saved runs sorted by last_updated descending."""
    if not RUNS_DIR.exists():
        return []

    runs = []
    for p in RUNS_DIR.glob("*.json"):
        try:
            envelope = TaskEnvelope.model_validate_json(p.read_text())
            last_updated = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            runs.append({
                "id": envelope.id,
                "goal": envelope.goal,
                "steps": len(envelope.steps_taken),
                "last_updated": last_updated,
            })
        except Exception:
            continue

    return sorted(runs, key=lambda r: r["last_updated"], reverse=True)


def delete(envelope_id: str) -> bool:
    """Delete a saved run. Returns True if it existed."""
    p = _path(envelope_id)
    if p.exists():
        p.unlink()
        return True
    return False


def clear() -> int:
    """Delete all saved runs. Returns number of files deleted."""
    if not RUNS_DIR.exists():
        return 0
    count = 0
    for p in RUNS_DIR.glob("*.json"):
        p.unlink()
        count += 1
    return count
