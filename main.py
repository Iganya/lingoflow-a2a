# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

from core.a2a_models import JSONRPCRequest, JSONRPCResponse, TaskResult
from core.translation_agent import LingoFlowAgent

load_dotenv()

agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = LingoFlowAgent()
    yield

app = FastAPI(
    title="LingoFlow A2A Translator",
    description="Multilingual translation agent with Telex A2A protocol",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/a2a/agent/translate")
async def a2a_endpoint(request: Request):
    try:
        body = await request.json()

        if body.get("jsonrpc") != "2.0" or "id" not in body:
            return JSONResponse(status_code=400, content={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32600, "message": "Invalid Request"}
            })

        rpc = JSONRPCRequest(**body)
        messages = []
        context_id = task_id = None
        config = None

        if rpc.method == "message/send":
            messages = [rpc.params.message]
            config = rpc.params.configuration
        elif rpc.method == "execute":
            messages = rpc.params.messages
            context_id = rpc.params.contextId
            task_id = rpc.params.taskId

        result = await agent.process_messages(
            messages=messages,
            context_id=context_id,
            task_id=task_id,
            config=config
        )

        return JSONRPCResponse(id=rpc.id, result=result).model_dump()

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {"code": -32603, "message": "Internal error", "data": {"details": str(e)}}
        })

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "lingoflow"}