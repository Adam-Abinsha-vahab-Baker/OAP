from __future__ import annotations
import httpx
from oap.envelope import TaskEnvelope


class HTTPTransport:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def invoke(self, envelope: TaskEnvelope) -> dict:
        payload = envelope.model_dump(mode="json")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/invoke",
                json=payload,
            )
            response.raise_for_status()
            return response.json()