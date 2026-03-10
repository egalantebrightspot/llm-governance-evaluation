"""
Router for generation endpoints.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from agents.generator_azure import generate_structured_output
from agents.validator import load_schema


router = APIRouter()


class GenerateRequest(BaseModel):
    task: str
    schema_name: str


class GenerateResponse(BaseModel):
    output: Dict[str, Any]


@router.post("", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Call the Generator agent to produce structured JSON for a task.
    """
    schema = load_schema(req.schema_name)
    output = generate_structured_output(req.task, schema)
    return GenerateResponse(output=output)

