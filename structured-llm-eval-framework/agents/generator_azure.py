"""
Azure OpenAI‑backed generator agent.

This module exposes a single public function:

    generate_structured_output(task, schema)

which sends the task to Azure OpenAI and returns a JSON object that
is intended to comply with the provided JSON Schema.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential


load_dotenv()  # Load variables from .env if present


def _get_azure_client() -> OpenAIClient:
    """
    Construct an Azure OpenAI client from environment variables.

    Required env vars (see .env.example):
      - AZURE_OPENAI_KEY
      - AZURE_OPENAI_ENDPOINT
    """
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        raise RuntimeError(
            "Azure OpenAI environment variables are not configured. "
            "Set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in your .env file."
        )

    return OpenAIClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))


def generate_structured_output(task: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured JSON response from Azure OpenAI for the given task.

    Parameters
    ----------
    task:
        Natural‑language description of the task or prompt for the model.
    schema:
        A JSON‑serializable dictionary representing the JSON Schema that the
        model output should follow.

    Returns
    -------
    dict
        Parsed JSON object returned by the model.

    Raises
    ------
    RuntimeError
        If Azure OpenAI is not configured correctly or the model reply is empty.
    ValueError
        If the model response cannot be parsed as JSON.
    """
    client = _get_azure_client()

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError(
            "AZURE_OPENAI_DEPLOYMENT is not set. Configure it in your .env file."
        )

    # We use strong instructions in the system message to enforce JSON output
    # that complies with the provided schema. Later, the validator will check
    # actual compliance.
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

    response = client.get_chat_completions(
        model=deployment,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    if not response.choices:
        raise RuntimeError("Azure OpenAI returned no choices for the request.")

    message = response.choices[0].message
    content = (message.content or "").strip() if message else ""

    if not content:
        raise RuntimeError("Azure OpenAI returned an empty response.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc}") from exc