from __future__ import annotations
import os
import httpx
from oap.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.base_url = os.environ.get("OAP_OLLAMA_URL", "http://localhost:11434")

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                client.get(f"{self.base_url}/api/tags")
            return True
        except Exception:
            return False

    async def complete(self, prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            data = response.json()
            return data["response"].strip()
