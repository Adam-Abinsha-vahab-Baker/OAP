import pytest
from unittest.mock import patch, MagicMock
from oap.llm.bedrock import BedrockProvider


def test_is_available_no_credentials():
    with patch("boto3.session.Session") as mock_session:
        mock_session.return_value.get_credentials.return_value = None
        assert BedrockProvider().is_available() is False


def test_is_available_no_boto3():
    with patch.dict("sys.modules", {"boto3": None}):
        assert BedrockProvider().is_available() is False


@pytest.mark.asyncio
async def test_complete():
    import json
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({
        "content": [{"text": "research-agent"}]
    }).encode()

    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": mock_body}

    with patch("boto3.session.Session") as mock_session:
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        provider = BedrockProvider()
        result = await provider.complete("pick an agent")
        assert result == "research-agent"
