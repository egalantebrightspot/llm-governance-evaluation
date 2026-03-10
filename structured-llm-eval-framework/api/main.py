"""
Entry point for the FastAPI application.

This module is intentionally kept thin: it wires up the FastAPI app,
routers, middleware, and lifecycle hooks, but does not contain any
business logic.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import generate, validate, evaluate, score, run


API_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan hook.

    Use this to initialize shared resources at startup (for example,
    preloading schemas, golden sets, or expensive client objects) and
    to clean them up on shutdown. Kept generic here so the details live
    in dedicated modules, not in main.py.
    """
    # Startup: initialize shared resources here if needed.
    yield
    # Shutdown: release resources here if needed.


app = FastAPI(
    title="Structured LLM Evaluation Framework",
    version=API_VERSION,
    description=(
        "A schema‑driven, Azure OpenAI–powered system for generating, "
        "validating, critiquing, and scoring structured LLM outputs."
    ),
    contact={
        "name": "LLM Governance Evaluation",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# CORS configuration – allow broad access for development; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
async def health() -> Dict[str, str]:
    """
    Lightweight health/readiness probe for orchestrators and monitors.
    """
    return {"status": "ok"}


app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(validate.router, prefix="/validate", tags=["validate"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluate"])
app.include_router(score.router, prefix="/score", tags=["score"])
app.include_router(run.router, prefix="/run", tags=["run"])

