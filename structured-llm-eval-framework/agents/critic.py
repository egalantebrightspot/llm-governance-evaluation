"""
Gold‑standard comparison critic agent.

This agent compares a model's JSON output to a golden example and
produces a structured critique report capturing:

  - correctness (field‑by‑field equality),
  - omissions (fields missing vs. the golden example),
  - hallucinations (extra fields not present in the golden example),
  - mismatches (same field present in both but with different values).

Key public function:

    compare_to_golden(output_json, golden_json) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, List


def _compare_values(
    path: List[str],
    output_value: Any,
    golden_value: Any,
    mismatches: List[Dict[str, Any]],
) -> None:
    """
    Compare two values and record mismatches with their JSON path.
    """
    if output_value == golden_value:
        return

    mismatches.append(
        {
            "path": path,
            "golden": golden_value,
            "output": output_value,
        }
    )


def _diff_objects(
    output_json: Dict[str, Any],
    golden_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute omissions, hallucinations, and mismatches between two objects.
    """
    omissions: List[Dict[str, Any]] = []
    hallucinations: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []

    golden_keys = set(golden_json.keys())
    output_keys = set(output_json.keys())

    # Fields expected in golden but missing from output
    for key in sorted(golden_keys - output_keys):
        omissions.append(
            {
                "path": [key],
                "golden": golden_json[key],
            }
        )

    # Fields present in output but not in golden
    for key in sorted(output_keys - golden_keys):
        hallucinations.append(
            {
                "path": [key],
                "output": output_json[key],
            }
        )

    # Common fields: check for mismatches (shallow comparison)
    for key in sorted(golden_keys & output_keys):
        _compare_values(
            [key],
            output_json[key],
            golden_json[key],
            mismatches,
        )

    return {
        "omissions": omissions,
        "hallucinations": hallucinations,
        "mismatches": mismatches,
    }


def compare_to_golden(
    output_json: Dict[str, Any],
    golden_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare model output against a golden JSON example.

    Parameters
    ----------
    output_json:
        JSON‑serializable dict produced by the model.
    golden_json:
        JSON‑serializable dict representing the ideal / gold‑standard output.

    Returns
    -------
    dict
        A structured critique report:
        {
          "is_exact_match": bool,
          "omissions": [...],
          "hallucinations": [...],
          "mismatches": [...]
        }
    """
    if not isinstance(output_json, dict) or not isinstance(golden_json, dict):
        # For the initial version we expect top‑level objects.
        raise TypeError("Both output_json and golden_json must be JSON objects (dict).")

    diff = _diff_objects(output_json, golden_json)

    is_exact_match = (
        not diff["omissions"]
        and not diff["hallucinations"]
        and not diff["mismatches"]
    )

    return {
        "is_exact_match": is_exact_match,
        **diff,
    }

