import os
import pytest
import respx
import httpx
from oap.llm.openai import OpenAIProvider


def test_is_available_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIProvider().is_available() is True


def test_is_available_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OAP_OPENAI_API_KEY", raising=False)
    assert OpenAIProvider().is_available() is False


@pytest.mark.asyncio
@respx.mock
async def test_complete(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "research-agent"}}]
        })
    )
    provider = OpenAIProvider()
    result = await provider.complete("pick an agent")
    assert result == "research-agent"
