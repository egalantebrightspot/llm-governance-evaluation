"""
Router for the full pipeline execution (/run).
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from agents.validator import load_schema
from evaluation.golden_set import load_golden_example
from evaluation.pipeline import run_evaluation


router = APIRouter()


class RunRequest(BaseModel):
    task: str
    schema_name: str
    golden_name: str


class RunResponse(BaseModel):
    trace_id: str | None = None
    task: str
    mode: str | None = None
    config: Dict[str, Any] | None = None
    stages: Dict[str, Any] | None = None
    output: Dict[str, Any] | None = None
    validation: Dict[str, Any] | None = None
    critique: Dict[str, Any] | None = None
    scores: Dict[str, float] | None = None


@router.post("", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    """
    Execute the full generator → validator → critic → scorer pipeline.
    """
    schema = load_schema(req.schema_name)
    golden = load_golden_example(req.golden_name)
    result = run_evaluation(req.task, schema, golden)
    return RunResponse(**result)

