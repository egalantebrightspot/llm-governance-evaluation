"""
Orchestration of the multi‑stage evaluation pipeline.

This module wires together the four core agents:

  1. Generator (Azure OpenAI)
  2. Validator (JSON Schema compliance)
  3. Critic (gold‑standard comparison)
  4. Scorer (metric computation)

Key public function:

    run_evaluation(task, schema, golden_example) -> dict
"""

from __future__ import annotations

from typing import Any, Dict

from agents.generator_azure import generate_structured_output
from agents.validator import validate_json
from agents.critic import compare_to_golden
from agents.scorer import compute_scores


def run_evaluation(
    task: str,
    schema: Dict[str, Any],
    golden_example: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute the full evaluation pipeline for a single task.

    Stage 1: Generator
        - Calls Azure OpenAI to produce structured JSON.

    Stage 2: Validator
        - Validates the JSON against the provided schema.

    Stage 3: Critic
        - Compares the JSON to a golden example, detecting omissions,
          hallucinations, and mismatches.

    Stage 4: Scorer
        - Converts the validation and critique into numeric metrics.

    Returns
    -------
    dict
        Unified evaluation object:
        {
          "task": task,
          "output": { ...model JSON... },
          "validation": { ...schema validation report... },
          "critique": { ...gold-standard comparison... },
          "scores": { ...numeric metrics... }
        }
    """
    # 1. Generate
    output_json = generate_structured_output(task, schema)

    # 2. Validate
    validation_report = validate_json(output_json, schema)

    # 3. Critique
    critique_report = compare_to_golden(output_json, golden_example)

    # 4. Score
    scores = compute_scores(validation_report, critique_report)

    return {
        "task": task,
        "output": output_json,
        "validation": validation_report,
        "critique": critique_report,
        "scores": scores,
    }

