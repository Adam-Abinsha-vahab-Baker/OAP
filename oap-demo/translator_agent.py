"""
Translator agent demo — implements the OAP agent interface.

Endpoints:
  GET  /        → health + capability info (used by oap ping and oap register)
  POST /invoke  → receives a TaskEnvelope, returns result + optional handoff
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

AGENT_ID = "translator-agent"
CAPABILITIES = ["translate", "simplify", "explain"]
DESCRIPTION = (
    "I rewrite technical content for non-technical audiences. "
    "I use plain English, analogies and friendly language to make complex topics accessible to anyone."
)


@app.get("/")
async def info():
    return {
        "agent_id": AGENT_ID,
        "capabilities": CAPABILITIES,
        "description": DESCRIPTION,
        "status": "ok",
    }


class InvokeRequest(BaseModel):
    goal: str
    memory: dict = {}


@app.post("/invoke")
async def invoke(request: InvokeRequest):
    # TODO: replace stub with real Claude Bedrock call
    return {
        "result": f"[{AGENT_ID}] Simplified for general audience: {request.goal}",
        "memory": {"translated_by": AGENT_ID},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
