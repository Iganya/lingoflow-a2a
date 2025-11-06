from fastapi.responses import JSONResponse
from datetime import datetime
from uuid import uuid4

from core.a2a_models import (A2AMessage, MessagePart, TaskResult,
                             TaskStatus, Artifact, ArtifactMessagePart, JSONRPCResponse)




def _invalid_body_response(
        request_id: str = "", 
        error: str = "wrong text format: use single '' for inline quote or enter 'help' to get guide"
    ) -> JSONResponse:
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