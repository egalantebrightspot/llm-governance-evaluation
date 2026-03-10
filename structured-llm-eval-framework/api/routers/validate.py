"""
Router for validation endpoints.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from agents.validator import load_schema, validate_structured_output


router = APIRouter()


class ValidateRequest(BaseModel):
    data: Dict[str, Any]
    schema_name: str


class ValidateResponse(BaseModel):
    is_valid: bool
    errors: list[Dict[str, Any]]
    missing_fields: list[Dict[str, Any]]
    extra_fields: list[Dict[str, Any]]
    type_errors: list[Dict[str, Any]]
    structure_errors: list[Dict[str, Any]]


@router.post("", response_model=ValidateResponse)
def validate(req: ValidateRequest) -> ValidateResponse:
    """
    Validate a JSON object against a named schema.
    """
    schema = load_schema(req.schema_name)
    report = validate_structured_output(req.data, schema)
    return ValidateResponse(**report)

