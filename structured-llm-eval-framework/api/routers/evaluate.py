"""
Router for evaluation endpoints (critic + scorer).
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from agents.critic import compare_to_golden
from agents.scorer import compute_scores
from evaluation.golden_set import load_golden_example


router = APIRouter()


class EvaluateRequest(BaseModel):
    output: Dict[str, Any]
    golden_name: str


class EvaluateResponse(BaseModel):
    critique: Dict[str, Any]
    scores: Dict[str, float]


@router.post("", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """
    Compare model output to a golden example and compute scores.
    """
    golden = load_golden_example(req.golden_name)
    critique = compare_to_golden(req.output, golden)
    scores = compute_scores(
        validation_report={"is_valid": True, "errors": []},
        critique_report=critique,
    )
    return EvaluateResponse(critique=critique, scores=scores)

