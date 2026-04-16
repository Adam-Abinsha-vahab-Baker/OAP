import pytest
import respx
import httpx
from oap.llm.ollama import OllamaProvider


def test_is_available_when_down(monkeypatch):
    monkeypatch.delenv("OAP_OLLAMA_URL", raising=False)
    with respx.mock:
        respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("down"))
        assert OllamaProvider().is_available() is False


@pytest.mark.asyncio
@respx.mock
async def test_complete():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "summarise-agent"})
    )
    provider = OllamaProvider()
    result = await provider.complete("summarise this")
    assert result == "summarise-agent"
