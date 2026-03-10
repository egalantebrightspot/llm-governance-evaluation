"""
Helpers for loading and managing golden‑set examples.

Golden examples are stored as JSON files under an evaluation/golden/
directory (not yet created by default). This module provides thin
helpers for loading those examples by name so that agents, such as
the Critic, can compare model outputs against them.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


GOLDEN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "golden",
)


def load_golden_example(name: str) -> Dict[str, Any]:
    """
    Load a golden JSON example by name.

    Parameters
    ----------
    name:
        Base name of the golden example file without extension
        (e.g. \"sample_ticket\", \"example_1\").

    Returns
    -------
    dict
        Parsed JSON object representing the golden example.

    Raises
    ------
    FileNotFoundError
        If the golden file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    filename = f"{name}.json" if not name.endswith(".json") else name
    path = os.path.join(GOLDEN_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

