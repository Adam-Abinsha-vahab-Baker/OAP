"""
Research agent demo — implements the OAP agent interface.

Endpoints:
  GET  /        → health + capability info (used by oap ping and oap register)
  POST /invoke  → receives a TaskEnvelope, returns result + optional handoff
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

AGENT_ID = "research-agent"
CAPABILITIES = ["research", "find", "search", "analyse", "compare"]
DESCRIPTION = (
    "I research topics thoroughly, find information from multiple angles, "
    "analyse options and summarise findings across any domain."
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
        "result": f"[{AGENT_ID}] Researched: {request.goal}",
        "memory": {"researched_by": AGENT_ID},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
