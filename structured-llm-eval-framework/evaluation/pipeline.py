"""
Orchestration of the multi‑stage evaluation pipeline.

This module wires together the four core agents:

  1. Generator (Azure OpenAI)
  2. Validator (JSON Schema compliance)
  3. Critic (gold‑standard comparison)
  4. Scorer (metric computation)

Key public functions:

    evaluate(...)
    run_evaluation(...)  # backwards‑compatible alias
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from agents.generator_azure import generate_structured_output
from agents.validator import validate_structured_output
from agents.critic import compare_to_golden
from agents.scorer import compute_scores


logger = logging.getLogger(__name__)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def evaluate(
    task: str,
    schema: dict[str, Any],
    golden_example: dict[str, Any],
    *,
    mode: str = "strict",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    """
    Execute the full evaluation pipeline for a single task.

    Stages
    ------
    1. Generator
       - Calls Azure OpenAI to produce structured JSON.

    2. Validator
       - Validates the JSON against the provided schema.

    3. Critic
       - Compares the JSON to a golden example, detecting omissions,
         hallucinations, and mismatches.

    4. Scorer
       - Converts the validation and critique into numeric metrics.

    Error handling
    --------------
    Each stage returns structured status and timing information. If a
    stage fails, its error is captured in the corresponding stage block,
    and later stages may be skipped. The function always returns a
    best‑effort partial result instead of raising directly.

    Parameters
    ----------
    mode:
        Evaluation mode, e.g. "strict" or "permissive". Currently
        informational but reserved for future behavior changes.
    temperature, max_tokens, top_p:
        Optional sampling parameters forwarded to the generator.
    """
    trace_id = uuid.uuid4().hex

    # Basic input validation.
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non‑empty string.")
    if not isinstance(schema, dict):
        raise TypeError("schema must be a dict.")
    if not isinstance(golden_example, dict):
        raise TypeError("golden_example must be a dict.")

    pipeline_result: dict[str, Any] = {
        "trace_id": trace_id,
        "task": task,
        "mode": mode,
        "config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        },
        "stages": {},
        "output": None,
        "validation": None,
        "critique": None,
        "scores": None,
    }

    # --- Stage 1: Generator ---
    gen_start = _now_ms()
    try:
        logger.info(
            "pipeline.generator.start",
            extra={"trace_id": trace_id, "mode": mode},
        )
        output_json = generate_structured_output(
            task,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        gen_end = _now_ms()
        pipeline_result["output"] = output_json
        pipeline_result["stages"]["generator"] = {
            "status": "success",
            "duration_ms": gen_end - gen_start,
        }
    except Exception as exc:
        gen_end = _now_ms()
        logger.error(
            "pipeline.generator.error",
            extra={"trace_id": trace_id, "mode": mode, "error": str(exc)},
        )
        pipeline_result["stages"]["generator"] = {
            "status": "error",
            "duration_ms": gen_end - gen_start,
            "error": str(exc),
        }
        # If we cannot generate, later stages cannot proceed.
        return pipeline_result

    # --- Stage 2: Validator ---
    val_start = _now_ms()
    try:
        logger.info(
            "pipeline.validator.start",
            extra={"trace_id": trace_id, "mode": mode},
        )
        validation_report = validate_structured_output(output_json, schema)
        val_end = _now_ms()
        pipeline_result["validation"] = validation_report
        pipeline_result["stages"]["validator"] = {
            "status": "success",
            "duration_ms": val_end - val_start,
        }
    except Exception as exc:
        val_end = _now_ms()
        logger.error(
            "pipeline.validator.error",
            extra={"trace_id": trace_id, "mode": mode, "error": str(exc)},
        )
        pipeline_result["stages"]["validator"] = {
            "status": "error",
            "duration_ms": val_end - val_start,
            "error": str(exc),
        }
        # Continue: downstream stages may still provide useful signals.

    # --- Stage 3: Critic ---
    crit_start = _now_ms()
    try:
        logger.info(
            "pipeline.critic.start",
            extra={"trace_id": trace_id, "mode": mode},
        )
        critique_report = compare_to_golden(output_json, golden_example)
        crit_end = _now_ms()
        pipeline_result["critique"] = critique_report
        pipeline_result["stages"]["critic"] = {
            "status": "success",
            "duration_ms": crit_end - crit_start,
        }
    except Exception as exc:
        crit_end = _now_ms()
        logger.error(
            "pipeline.critic.error",
            extra={"trace_id": trace_id, "mode": mode, "error": str(exc)},
        )
        pipeline_result["stages"]["critic"] = {
            "status": "error",
            "duration_ms": crit_end - crit_start,
            "error": str(exc),
        }
        # Without a critique we cannot compute scores, but still return
        # generation + validation information.
        return pipeline_result

    # --- Stage 4: Scorer ---
    score_start = _now_ms()
    try:
        logger.info(
            "pipeline.scorer.start",
            extra={"trace_id": trace_id, "mode": mode},
        )
        validation_for_scoring = (
            pipeline_result["validation"]
            if isinstance(pipeline_result["validation"], dict)
            else {"is_valid": False, "errors": []}
        )
        scores = compute_scores(validation_for_scoring, critique_report)
        score_end = _now_ms()
        pipeline_result["scores"] = scores
        pipeline_result["stages"]["scorer"] = {
            "status": "success",
            "duration_ms": score_end - score_start,
        }
    except Exception as exc:
        score_end = _now_ms()
        logger.error(
            "pipeline.scorer.error",
            extra={"trace_id": trace_id, "mode": mode, "error": str(exc)},
        )
        pipeline_result["stages"]["scorer"] = {
            "status": "error",
            "duration_ms": score_end - score_start,
            "error": str(exc),
        }

    return pipeline_result


def run_evaluation(
    task: str,
    schema: dict[str, Any],
    golden_example: dict[str, Any],
) -> dict[str, Any]:
    """
    Backwards‑compatible wrapper for evaluate().

    New code should call evaluate() directly.
    """
    return evaluate(task, schema, golden_example)

