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



def _invalid_body_response(request_id: str = "", error: str = "Unknown method. Use 'message/send' or 'help'.") -> JSONResponse:
    """
    Returns the exact 200-OK response you specified when the body is empty.
    """
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    task_id = str(uuid4())
    context_id = str(uuid4())
    message_id = str(uuid4())
    artifact_id = str(uuid4())

    message = A2AMessage(
        kind="message",
        role="agent",
        parts=[MessagePart(kind="text", text=error)],
        messageId=message_id,
        taskId=None,
        metadata=None,
    )

    status = TaskStatus(state="failed", timestamp=now, message=message)

    artifact = Artifact(
        artifactId=artifact_id,
        name="assistantResponse",
        parts=[ArtifactMessagePart(kind="text", text=error)],
    )

    result = TaskResult(
        id=task_id,
        contextId=context_id,
        status=status,
        artifacts=[artifact],
        history=[],
        kind="task",
    )

    resp = JSONRPCResponse(id=request_id, result=result)
    return JSONResponse(content=resp.model_dump(), status_code=200)



@app.post("/a2a/lingflow")
async def a2a_endpoint(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        # Not even valid JSON → treat as empty
        return _invalid_body_response(request_id="", error="Unknown method. Use 'message/send' or 'help'.")

    if not body:
        return _invalid_body_response(request_id="", error="Unknown method. Use 'message/send' or 'help'.")

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

        return JSONRPCResponse(id=rpc.id, result=result).model_dump()

    except Exception as e:
        return _invalid_body_response(request_id="", error=str(e))

    


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "lingoflow"}