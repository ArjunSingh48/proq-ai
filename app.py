from __future__ import annotations

import os
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from request_workflow import RequestWorkflowService
from supplier_engine import SupplierEngine


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(Path(__file__).parent / ".env")

app = FastAPI()
engine = SupplierEngine()
workflow_service = RequestWorkflowService(engine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    request: dict[str, Any] = Field(...)


class ChatWorkflowRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


@app.post("/match")
def match(payload: MatchRequest) -> dict[str, Any]:
    request_dict = payload.request
    request_id = request_dict.get("request_id", "unknown")

    print(f"[match] start request_id={request_id}")
    result = engine.process(request_dict)
    print(f"[match] end request_id={request_id}")

    return result


@app.post("/workflow")
def workflow(payload: ChatWorkflowRequest) -> dict[str, Any]:
    try:
        result = workflow_service.run(payload.message.strip(), payload.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    request_id = result["request"]["request_id"]

    print(f"[workflow] status={result['status']} request_id={request_id} parser={result['parser_source']} session_id={result['session_id']}")
    return result
