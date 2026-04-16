from __future__ import annotations
from oap.llm.base import LLMProvider


def get_provider(config: dict | None = None) -> LLMProvider | None:
    """
    Returns the configured LLM provider or None if not configured.
    Priority:
    1. Explicit config dict passed in
    2. ~/.oap/config.json
    3. None (falls back to keyword matching)
    """
    if config is None:
        from oap.config import get_llm_config
        config = get_llm_config()

    if config is None:
        return None

    provider_name = config.get("provider")
    model = config.get("model")

    if provider_name == "bedrock":
        from oap.llm.bedrock import BedrockProvider
        kwargs = {}
        if model:
            kwargs["model"] = model
        return BedrockProvider(**kwargs)
    elif provider_name == "openai":
        from oap.llm.openai import OpenAIProvider
        kwargs = {}
        if model:
            kwargs["model"] = model
        return OpenAIProvider(**kwargs)
    elif provider_name == "ollama":
        from oap.llm.ollama import OllamaProvider
        kwargs = {}
        if model:
            kwargs["model"] = model
        return OllamaProvider(**kwargs)
    elif provider_name == "custom":
        from oap.llm.custom import CustomProvider
        kwargs = {}
        if model:
            kwargs["model"] = model
        return CustomProvider(**kwargs)

    return None
