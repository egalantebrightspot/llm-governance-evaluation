"""
Router for scoring endpoints.

This endpoint focuses purely on converting an existing validation
and critique report into numeric metrics.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from agents.scorer import compute_scores


router = APIRouter()


class ScoreRequest(BaseModel):
    validation: Dict[str, Any]
    critique: Dict[str, Any]


class ScoreResponse(BaseModel):
    scores: Dict[str, float]


@router.post("", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """
    Compute scores from pre‑computed validation and critique reports.
    """
    scores = compute_scores(req.validation, req.critique)
    return ScoreResponse(scores=scores)

