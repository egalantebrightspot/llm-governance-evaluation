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

from typing import Any


def _safe_len(obj: Any) -> int:
    try:
        return len(obj)
    except TypeError:
        return 0


def compute_scores(
    validation_report: dict[str, Any],
    critique_report: dict[str, Any],
) -> dict[str, float]:
    """
    Compute numeric scores from validation and critique reports.

    Metrics
    -------
    - schema_compliance (0–1):
        Soft score based on the number of validation issues. 1.0 when there
        are no issues, decaying towards 0.0 as issues increase.

    - field_accuracy (0–1):
        Proportion of golden fields that are present AND correct.
        Uses explicit correct_fields / field counts from the critic when
        available, and falls back to simple estimates otherwise.

    - hallucination_rate (0–1):
        Fraction of output fields that are hallucinations, using the total
        number of output fields (including correct ones) as denominator.

    - completeness (0–1):
        1 - (omissions / max(golden_fields, 1))

    Returns
    -------
    dict with float scores in [0, 1].
    """
    # --- Schema compliance ---
    is_valid = bool(validation_report.get("is_valid"))
    validation_errors = validation_report.get("errors") or []
    issue_count = _safe_len(validation_errors)

    if not is_valid:
        # Soft decay: 1.0 with 0 issues, ~0.5 at 10 issues, then towards 0.
        schema_compliance = 1.0 - min(1.0, issue_count / 20.0)
    else:
        schema_compliance = 1.0

    # --- Field‑level signals from critic ---
    omissions = critique_report.get("omissions") or []
    hallucinations = critique_report.get("hallucinations") or []
    mismatches = critique_report.get("mismatches") or []

    num_omissions = _safe_len(omissions)
    num_hallucinations = _safe_len(hallucinations)
    num_mismatches = _safe_len(mismatches)

    # Prefer explicit counts / correct fields if the critic provides them.
    explicit_correct_fields = critique_report.get("correct_fields")
    if isinstance(explicit_correct_fields, list):
        num_correct = _safe_len(explicit_correct_fields)
    else:
        num_correct = 0

    golden_field_count = critique_report.get("golden_field_count")
    output_field_count = critique_report.get("output_field_count")

    if isinstance(golden_field_count, int) and golden_field_count >= 0:
        golden_total = golden_field_count
    else:
        golden_total = num_omissions + num_mismatches + num_correct

    if isinstance(output_field_count, int) and output_field_count >= 0:
        output_total = output_field_count
    else:
        output_total = num_hallucinations + num_mismatches + num_correct

    denom_golden = max(golden_total, 1)
    denom_output = max(output_total, 1)

    field_accuracy = (num_correct / denom_golden) if denom_golden else 0.0
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

