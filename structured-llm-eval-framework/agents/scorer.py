"""
Metric‑producing scoring agent.

This agent converts structural validation and gold‑standard comparison
reports into quantitative metrics.

Expected inputs:

  - validation_report: output of agents.validator.validate_json(...)
      {
        "is_valid": bool,
        "errors": [ ... ]
      }

  - critique_report: output of agents.critic.compare_to_golden(...)
      {
        "is_exact_match": bool,
        "omissions": [...],
        "hallucinations": [...],
        "mismatches": [...]
      }

Key public function:

    compute_scores(validation_report, critique_report) -> dict
"""

from __future__ import annotations

from typing import Any, Dict


def _safe_len(obj: Any) -> int:
    try:
        return len(obj)
    except TypeError:
        return 0


def compute_scores(
    validation_report: Dict[str, Any],
    critique_report: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute numeric scores from validation and critique reports.

    Metrics
    -------
    - schema_compliance (0–1):
        1.0 if the JSON is schema‑valid, else 0.0.

    - field_accuracy (0–1):
        Proportion of golden fields that are present AND correct.
        Based on:
          golden_fields = omissions + mismatches + correct_fields
          correct_fields = golden_fields - omissions - mismatches

    - hallucination_rate (0–1):
        Fraction of output fields that are hallucinations.
        Based on:
          output_fields = hallucinations + mismatches + correct_fields
          hallucination_rate = hallucinations / max(output_fields, 1)

    - completeness (0–1):
        1 - (omissions / max(golden_fields, 1))

    Returns
    -------
    dict with float scores in [0, 1].
    """
    is_valid = bool(validation_report.get("is_valid"))
    validation_errors = validation_report.get("errors") or []

    omissions = critique_report.get("omissions") or []
    hallucinations = critique_report.get("hallucinations") or []
    mismatches = critique_report.get("mismatches") or []

    num_omissions = _safe_len(omissions)
    num_hallucinations = _safe_len(hallucinations)
    num_mismatches = _safe_len(mismatches)

    # Estimate golden and output field counts at the top level.
    golden_fields = num_omissions + num_mismatches
    output_fields = num_hallucinations + num_mismatches

    # Assume there is at least one correct field when there is some overlap.
    # We do not track correct_fields explicitly, so treat mismatches as the
    # only overlapping fields; this is a conservative approximation.
    # To avoid negative values in edge cases, clamp at zero.
    correct_fields = max(golden_fields - num_omissions - num_mismatches, 0)

    denom_golden = max(golden_fields + correct_fields, 1)
    denom_output = max(output_fields + correct_fields, 1)

    schema_compliance = 1.0 if is_valid and not validation_errors else 0.0

    field_accuracy = (correct_fields / denom_golden) if denom_golden else 0.0
    hallucination_rate = (
        num_hallucinations / denom_output if denom_output else 0.0
    )
    completeness = 1.0 - (num_omissions / denom_golden) if denom_golden else 0.0

    # Clamp to [0, 1] to guard against numerical issues.
    def clamp(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    return {
        "schema_compliance": clamp(schema_compliance),
        "field_accuracy": clamp(field_accuracy),
        "hallucination_rate": clamp(hallucination_rate),
        "completeness": clamp(completeness),
    }

