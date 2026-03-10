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
    task: str
    output: Dict[str, Any]
    validation: Dict[str, Any]
    critique: Dict[str, Any]
    scores: Dict[str, float]


@router.post("", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    """
    Execute the full generator → validator → critic → scorer pipeline.
    """
    schema = load_schema(req.schema_name)
    golden = load_golden_example(req.golden_name)
    result = run_evaluation(req.task, schema, golden)
    return RunResponse(**result)

