from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from uuid import uuid4
from core.a2a_models import (JSONRPCRequest, JSONRPCResponse, TaskResult, ArtifactMessagePart,
                             A2AMessage, MessagePart, Artifact, TaskStatus)
from core.translation_agent import LingoFlowAgent
from core.response import _invalid_body_response

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



@app.post("/a2a/lingflow")
async def a2a_endpoint(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        # Not even valid JSON → treat as empty
        return _invalid_body_response(request_id="", error="wrong text format: use single '' for inline quote or enter 'help' to get guide")

    if not body:
        return _invalid_body_response(request_id="", error="wrong text format: use single '' for inline quote or enter 'help' to get guide")

    #  Basic JSON-RPC validation 
    if body.get("jsonrpc") != "2.0" or "id" not in body:
        return _invalid_body_response(request_id="", error="Invalid Request: jsonrpc must be '2.0' and id is required")
        
    try:
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
            config=config,
        )
        try:
            return JSONRPCResponse(id=rpc.id, result=result).model_dump()
        except Exception as e:
            return result
    except Exception as e:
        return _invalid_body_response(request_id="", error=f"{e} or enter 'help' to get standard guide")

    


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "lingoflow"}