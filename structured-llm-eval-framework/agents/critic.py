"""
Gold‑standard comparison critic agent.

This agent compares a model's JSON output to a golden example and
produces a structured critique report capturing:

  - correctness (field‑by‑field equality, including nested structures),
  - omissions (fields missing vs. the golden example),
  - hallucinations (extra fields not present in the golden example),
  - mismatches (same field present in both but with different values or types).

Key public function:

    compare_to_golden(output_json, golden_json, schema=None) -> dict
"""

from __future__ import annotations

from typing import Any


def _type_name(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def _make_issue(
    *,
    path: list[Any],
    code: str,
    message: str,
    golden: Any | None = None,
    output: Any | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "path": path,
        "golden": golden,
        "output": output,
        "expected_type": _type_name(golden),
        "actual_type": _type_name(output),
    }


def _diff_values(
    path: list[Any],
    output_value: Any,
    golden_value: Any,
    *,
    omissions: list[dict[str, Any]],
    hallucinations: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
    correct_fields: list[list[Any]],
) -> None:
    """
    Recursively compare two JSON values, recording issues and correct leaves.
    """
    # Exact equality – treat as a correct leaf (for scoring hints).
    if output_value == golden_value:
        correct_fields.append(list(path))
        return

    # Both dicts: compare keys recursively.
    if isinstance(output_value, dict) and isinstance(golden_value, dict):
        out_keys = set(output_value.keys())
        gold_keys = set(golden_value.keys())

        for key in sorted(gold_keys - out_keys):
            omissions.append(
                _make_issue(
                    path=path + [key],
                    code="omission",
                    message=f"Missing field '{key}' compared to golden.",
                    golden=golden_value.get(key),
                    output=None,
                )
            )

        for key in sorted(out_keys - gold_keys):
            hallucinations.append(
                _make_issue(
                    path=path + [key],
                    code="hallucination",
                    message=f"Extra field '{key}' not present in golden.",
                    golden=None,
                    output=output_value.get(key),
                )
            )

        for key in sorted(out_keys & gold_keys):
            _diff_values(
                path + [key],
                output_value[key],
                golden_value[key],
                omissions=omissions,
                hallucinations=hallucinations,
                mismatches=mismatches,
                correct_fields=correct_fields,
            )
        return

    # Both lists: compare element‑wise (order‑sensitive for now).
    if isinstance(output_value, list) and isinstance(golden_value, list):
        min_len = min(len(output_value), len(golden_value))
        for idx in range(min_len):
            _diff_values(
                path + [idx],
                output_value[idx],
                golden_value[idx],
                omissions=omissions,
                hallucinations=hallucinations,
                mismatches=mismatches,
                correct_fields=correct_fields,
            )

        # Extra elements in golden → omissions.
        for idx in range(min_len, len(golden_value)):
            omissions.append(
                _make_issue(
                    path=path + [idx],
                    code="omission",
                    message="Missing array element compared to golden.",
                    golden=golden_value[idx],
                    output=None,
                )
            )

        # Extra elements in output → hallucinations.
        for idx in range(min_len, len(output_value)):
            hallucinations.append(
                _make_issue(
                    path=path + [idx],
                    code="hallucination",
                    message="Extra array element not present in golden.",
                    golden=None,
                    output=output_value[idx],
                )
            )
        return

    # Type mismatch vs value mismatch.
    if _type_name(output_value) != _type_name(golden_value):
        mismatches.append(
            _make_issue(
                path=path,
                code="type_mismatch",
                message="Type mismatch between output and golden.",
                golden=golden_value,
                output=output_value,
            )
        )
    else:
        mismatches.append(
            _make_issue(
                path=path,
                code="value_mismatch",
                message="Value differs between output and golden.",
                golden=golden_value,
                output=output_value,
            )
        )


def compare_to_golden(
    output_json: dict[str, Any],
    golden_json: dict[str, Any],
    schema: dict[str, Any] | None = None,  # reserved for future schema‑aware diff
) -> dict[str, Any]:
    """
    Compare model output against a golden JSON example.

    Parameters
    ----------
    output_json:
        JSON‑serializable dict produced by the model.
    golden_json:
        JSON‑serializable dict representing the ideal / gold‑standard output.
    schema:
        Optional JSON Schema fragment describing the structure. Currently
        unused but reserved for future schema‑aware comparison behaviors.

    Returns
    -------
    dict
        A structured critique report:
        {
          "is_exact_match": bool,
          "omissions": [...],
          "hallucinations": [...],
          "mismatches": [...],
          "correct_fields": [...],
          "golden_field_count": int,
          "output_field_count": int
        }
    """
    if not isinstance(output_json, dict) or not isinstance(golden_json, dict):
        raise TypeError("Both output_json and golden_json must be JSON objects (dict).")

    omissions: list[dict[str, Any]] = []
    hallucinations: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    correct_fields: list[list[Any]] = []

    _diff_values(
        [],
        output_json,
        golden_json,
        omissions=omissions,
        hallucinations=hallucinations,
        mismatches=mismatches,
        correct_fields=correct_fields,
    )

    is_exact_match = not omissions and not hallucinations and not mismatches

    # Approximate field counts using number of leaf paths we touched.
    golden_field_count = len(correct_fields) + len(omissions) + len(
        [m for m in mismatches if m["golden"] is not None]
    )
    output_field_count = len(correct_fields) + len(hallucinations) + len(
        [m for m in mismatches if m["output"] is not None]
    )

    return {
        "is_exact_match": is_exact_match,
        "omissions": omissions,
        "hallucinations": hallucinations,
        "mismatches": mismatches,
        "correct_fields": correct_fields,
        "golden_field_count": golden_field_count,
        "output_field_count": output_field_count,
    }

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

