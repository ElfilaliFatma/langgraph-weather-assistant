"""Tolerant JSON extraction."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(raw)] + [raw]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    for candidate in candidates:
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate.strip()))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None
