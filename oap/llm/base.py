from __future__ import annotations
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Send prompt, return text response."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured."""
        ...
