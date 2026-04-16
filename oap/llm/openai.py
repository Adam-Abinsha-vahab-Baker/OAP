from __future__ import annotations
import os
import httpx
from oap.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OAP_OPENAI_API_KEY"))

    async def complete(self, prompt: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OAP_OPENAI_API_KEY")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
