"""
Entry point for the FastAPI application.

Exposes routers for:
  - /generate
  - /validate
  - /evaluate
  - /score
  - /run (full pipeline)
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routers import generate, validate, evaluate, score, run


app = FastAPI(title="Structured LLM Evaluation Framework")

app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(validate.router, prefix="/validate", tags=["validate"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluate"])
app.include_router(score.router, prefix="/score", tags=["score"])
app.include_router(run.router, prefix="/run", tags=["run"])

