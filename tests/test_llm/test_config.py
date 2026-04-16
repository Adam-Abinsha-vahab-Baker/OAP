import pytest
from pathlib import Path
import json
from unittest.mock import patch
import oap.config as config_module


@pytest.fixture(autouse=True)
def tmp_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_path)


def test_get_llm_config_returns_none_when_not_set():
    assert config_module.get_llm_config() is None


def test_set_and_get_llm_config():
    config_module.set_llm_config("openai", "gpt-4o-mini")
    cfg = config_module.get_llm_config()
    assert cfg["provider"] == "openai"
    assert cfg["model"] == "gpt-4o-mini"


def test_clear_llm_config():
    config_module.set_llm_config("openai")
    config_module.clear_llm_config()
    assert config_module.get_llm_config() is None


def test_get_provider_returns_none_when_no_config():
    from oap.llm.factory import get_provider
    assert get_provider(config={}) is None


def test_get_provider_returns_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from oap.llm.factory import get_provider
    from oap.llm.openai import OpenAIProvider
    provider = get_provider(config={"provider": "openai", "model": "gpt-4o-mini"})
    assert isinstance(provider, OpenAIProvider)
