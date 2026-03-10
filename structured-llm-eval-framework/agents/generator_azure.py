"""
Azure OpenAI‑backed generator agent.

This module exposes a single public function:

    generate_structured_output(task, schema)

which sends the task to Azure OpenAI and returns a JSON object that
is intended to comply with the provided JSON Schema.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAIError
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from jsonschema import Draft7Validator


load_dotenv()  # Load variables from .env if present

logger = logging.getLogger(__name__)

_CLIENT: AzureOpenAI | None = None


def _get_azure_client() -> AzureOpenAI:
    """
    Construct (or retrieve) a cached Azure OpenAI client from env vars.

    Required env vars (see .env.example):
      - AZURE_OPENAI_KEY
      - AZURE_OPENAI_ENDPOINT
    """
    global _CLIENT

    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        raise RuntimeError(
            "Azure OpenAI environment variables are not configured. "
            "Set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in your .env file."
        )

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    _CLIENT = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    return _CLIENT


def _is_transient_error(error: Exception) -> bool:
    """
    Best‑effort detection of transient errors (e.g. 429/5xx).
    """
    if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        if error.status_code in {429, 500, 502, 503, 504}:
            return True
    return False


def _validate_against_schema_if_enabled(
    obj: dict[str, Any],
    schema: dict[str, Any],
    validate_schema: bool,
) -> None:
    """
    Optionally perform an inline jsonschema validation for debugging/tests.
    """
    if not validate_schema:
        return

    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(obj))
    if errors:
        messages = "; ".join(e.message for e in errors)
        raise ValueError(f"Inline schema validation failed: {messages}")


def generate_structured_output(
    task: str,
    schema: dict[str, Any],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float | None = None,
    validate_schema: bool = False,
) -> dict[str, Any]:
    """
    Generate a structured JSON response from Azure OpenAI for the given task.

    Parameters
    ----------
    task:
        Natural‑language description of the task or prompt for the model.
    schema:
        A JSON‑serializable dictionary representing the JSON Schema that the
        model output should follow.
    temperature, max_tokens, top_p:
        Optional sampling parameters passed through to the model.
    validate_schema:
        When True, run an inline jsonschema check before returning. This is
        intended for debugging/unit tests; the main validator still runs
        later in the pipeline.

    Returns
    -------
    dict
        Parsed JSON object returned by the model.

    Raises
    ------
    RuntimeError
        If Azure OpenAI is not configured correctly, the model reply is empty,
        or repeated transient failures occur.
    ValueError
        If the model response cannot be parsed as JSON, is not an object, or
        fails inline schema validation.
    """
    client = _get_azure_client()

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError(
            "AZURE_OPENAI_DEPLOYMENT is not set. Configure it in your .env file."
        )

    system_instructions = (
        "You are a JSON‑only assistant. "
        "You MUST respond with a single valid JSON object that strictly follows "
        "the provided JSON Schema. Do not include any explanatory text, "
        "comments, or Markdown—only JSON."
    )

    user_prompt = (
        "Task:\n"
        f"{task}\n\n"
        "JSON Schema (for the response object):\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Return only a JSON object that conforms to this schema."
    )

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_prompt},
    ]

    max_attempts = 3
    backoff_seconds = 0.5
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        start = time.perf_counter()
        try:
            logger.info(
                "generator_azure.request",
                extra={
                    "model": deployment,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "attempt": attempt,
                },
            )

            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if not response.choices:
                raise RuntimeError("Azure OpenAI returned no choices for the request.")

            message = response.choices[0].message
            content = (message.content or "").strip() if message else ""

            logger.info(
                "generator_azure.response",
                extra={
                    "model": deployment,
                    "latency_ms": latency_ms,
                    "preview": content[:200],
                },
            )

            if not content:
                raise RuntimeError("Azure OpenAI returned an empty response.")

            try:
                obj = json.loads(content)
            except json.JSONDecodeError as exc:
                snippet = content[:200]
                logger.error(
                    "generator_azure.json_parse_error",
                    extra={"error": str(exc), "preview": snippet},
                )
                raise ValueError(
                    f"Model response is not valid JSON: {exc}. "
                    f"Preview: {snippet!r}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    "Model response must be a JSON object at the top level "
                    f"(got {type(obj).__name__})."
                )

            _validate_against_schema_if_enabled(obj, schema, validate_schema)

            return obj

        except OpenAIError as exc:
            last_error = exc
            is_transient = _is_transient_error(exc)
            logger.warning(
                "generator_azure.azure_error",
                extra={
                    "attempt": attempt,
                    "transient": is_transient,
                    "error": str(exc),
                },
            )
            if attempt == max_attempts or not is_transient:
                break
            time.sleep(backoff_seconds)
            backoff_seconds *= 2
        except Exception as exc:
            last_error = exc
            logger.error(
                "generator_azure.unexpected_error",
                extra={"attempt": attempt, "error": str(exc)},
            )
            break

    # If we get here, retries have been exhausted or error is non‑transient.
    if last_error:
        raise RuntimeError(f"Failed to generate structured output: {last_error}") from last_error
    raise RuntimeError("Failed to generate structured output for unknown reasons.")
