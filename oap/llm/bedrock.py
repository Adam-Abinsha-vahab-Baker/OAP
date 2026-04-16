from __future__ import annotations
import json
import os
from oap.llm.base import LLMProvider


class BedrockProvider(LLMProvider):
    def __init__(self, model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"):
        self.model = model
        self.region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    def is_available(self) -> bool:
        try:
            import boto3
            session = boto3.session.Session()
            creds = session.get_credentials()
            return creds is not None
        except (ImportError, Exception):
            return False

    async def complete(self, prompt: str) -> str:
        import boto3
        session = boto3.session.Session()
        client = session.client("bedrock-runtime", region_name=self.region)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
        })
        response = client.invoke_model(modelId=self.model, body=body)
        data = json.loads(response["body"].read())
        return data["content"][0]["text"]
