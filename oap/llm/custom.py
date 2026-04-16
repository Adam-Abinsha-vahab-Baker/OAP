from __future__ import annotations
import os
import httpx
from oap.llm.base import LLMProvider


class CustomProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OAP_CUSTOM_MODEL", "")
        self.base_url = os.environ.get("OAP_CUSTOM_BASE_URL", "")
        self.api_key = os.environ.get("OAP_CUSTOM_API_KEY")

    def is_available(self) -> bool:
        return bool(os.environ.get("OAP_CUSTOM_BASE_URL"))

    async def complete(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
