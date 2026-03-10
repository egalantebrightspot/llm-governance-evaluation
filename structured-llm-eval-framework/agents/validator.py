"""
Schema and hallucination validator agent.

This module is responsible for enforcing structural correctness of
model outputs against JSON Schemas.

Key public functions:

    load_schema(name) -> dict
    validate_json(data, schema) -> dict
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from jsonschema import Draft7Validator, ValidationError


SCHEMAS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas"
)


def load_schema(name: str) -> Dict[str, Any]:
    """
    Load a JSON Schema from the schemas directory.

    Parameters
    ----------
    name:
        Base name of the schema file without extension
        (e.g. \"classification\", \"extraction\", \"reasoning\").

    Returns
    -------
    dict
        Parsed JSON schema.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.
    json.JSONDecodeError
        If the schema file is not valid JSON.
    """
    filename = f"{name}.json" if not name.endswith(".json") else name
    path = os.path.join(SCHEMAS_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_error(err: ValidationError) -> Dict[str, Any]:
    """
    Convert a jsonschema.ValidationError into a serializable dict.
    """
    return {
        "message": err.message,
        "path": list(err.path),
        "schema_path": list(err.schema_path),
        "validator": err.validator,
        "validator_value": err.validator_value,
    }


def validate_json(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a JSON‑serializable object against a JSON Schema.

    Parameters
    ----------
    data:
        The JSON object produced by the generator.
    schema:
        The JSON Schema to validate against.

    Returns
    -------
    dict
        A structured validation report:
        {
          "is_valid": bool,
          "errors": [ { ...error details... }, ... ]
        }
    """
    validator = Draft7Validator(schema)
    errors: List[ValidationError] = list(validator.iter_errors(data))

    return {
        "is_valid": not errors,
        "errors": [_format_error(e) for e in errors],
    }

