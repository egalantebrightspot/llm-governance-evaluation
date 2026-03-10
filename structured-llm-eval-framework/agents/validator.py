"""
Schema and hallucination validator agent.

This module is responsible for enforcing structural correctness of
model outputs against JSON Schemas.

Key public functions:

    load_schema(name) -> dict
    validate_structured_output(data, schema) -> dict
"""

from __future__ import annotations

import json
import os
from typing import Any

from jsonschema import Draft7Validator, ValidationError


SCHEMAS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas"
)

_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def load_schema(name: str) -> dict[str, Any]:
    """
    Load a JSON Schema from the schemas directory, with caching.

    The loaded schema is validated as a Draft‑07 JSON Schema before
    being cached and returned.

    Parameters
    ----------
    name:
        Base name of the schema file without extension
        (e.g. "classification", "extraction", "reasoning").

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
    jsonschema.SchemaError
        If the loaded schema itself is not a valid Draft‑07 schema.
    """
    key = name.removesuffix(".json")
    if key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[key]

    filename = f"{key}.json"
    path = os.path.join(SCHEMAS_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        schema: dict[str, Any] = json.load(f)

    # Validate that the schema itself is a valid Draft‑07 schema.
    Draft7Validator.check_schema(schema)

    _SCHEMA_CACHE[key] = schema
    return schema


def _error_code(err: ValidationError) -> str:
    """
    Map jsonschema.ValidationError instances to stable error codes.
    """
    if err.validator == "required":
        return "missing_required_field"
    if err.validator == "type":
        return "type_mismatch"
    if err.validator in {"additionalProperties", "unevaluatedProperties"}:
        return "extra_field"
    if err.validator == "enum":
        return "invalid_enum_value"
    return "schema_violation"


def _format_error(err: ValidationError) -> dict[str, Any]:
    """
    Convert a jsonschema.ValidationError into a serializable dict.
    """
    return {
        "code": _error_code(err),
        "message": err.message,
        "path": list(err.path),
        "schema_path": list(err.schema_path),
        "validator": err.validator,
        "validator_value": err.validator_value,
        "instance": err.instance,
    }


def validate_structured_output(
    data: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate a JSON‑serializable object against a JSON Schema.

    This function is designed for model outputs and treats the top‑level
    value as expected to be a JSON object (dict).

    Returns
    -------
    dict
        A structured validation report:
        {
          "is_valid": bool,
          "errors": [ ... ],
          "missing_fields": [ ... ],
          "extra_fields": [ ... ],
          "type_errors": [ ... ],
          "structure_errors": [ ... ]
        }
    """
    # Guardrail: top‑level must be an object.
    if not isinstance(data, dict):
        error = {
            "code": "non_object_root",
            "message": "Top‑level JSON value must be an object.",
            "path": [],
            "schema_path": [],
            "validator": "type",
            "validator_value": "object",
            "instance": data,
        }
        return {
            "is_valid": False,
            "errors": [error],
            "missing_fields": [],
            "extra_fields": [],
            "type_errors": [error],
            "structure_errors": [error],
        }

    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(data))

    formatted_errors = [_format_error(e) for e in errors]

    missing_fields: list[dict[str, Any]] = []
    extra_fields: list[dict[str, Any]] = []
    type_errors: list[dict[str, Any]] = []
    structure_errors: list[dict[str, Any]] = []

    # Classify validation errors.
    for e in formatted_errors:
        code = e["code"]
        if code == "missing_required_field":
            missing_fields.append(e)
        elif code == "extra_field":
            extra_fields.append(e)
        elif code == "type_mismatch":
            type_errors.append(e)
        else:
            structure_errors.append(e)

    # Detect hallucinated fields w.r.t. schema properties, even if the
    # schema does not explicitly disable additionalProperties.
    props = schema.get("properties")
    if isinstance(props, dict):
        schema_fields = set(props.keys())
        data_fields = set(data.keys())
        hallucinated = sorted(data_fields - schema_fields)
        for field in hallucinated:
            record = {
                "code": "extra_field",
                "message": f"Field '{field}' is not defined in schema properties.",
                "path": [field],
                "schema_path": [],
                "validator": "properties",
                "validator_value": list(schema_fields),
                "instance": data.get(field),
            }
            extra_fields.append(record)
            structure_errors.append(record)

    is_valid = not formatted_errors and not extra_fields

    return {
        "is_valid": is_valid,
        "errors": formatted_errors + extra_fields,  # include synthesized ones
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "type_errors": type_errors,
        "structure_errors": structure_errors,
    }


def validate_json(data: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """
    Backwards‑compatible wrapper for validate_structured_output.

    Prefer validate_structured_output for new code.
    """
    return validate_structured_output(data, schema)

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

