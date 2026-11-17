"""Sanitize untrusted tool and web content before it reaches the model again."""

from __future__ import annotations

import re

_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions"),
    re.compile(r"(?i)system\s+prompt"),
    re.compile(r"(?i)you\s+are\s+now"),
    re.compile(r"(?i)disregard\s+.*(policy|rules)"),
    re.compile(r"(?i)run\s+shell|execute\s+command|os\.system"),
)


def sanitize_untrusted_text(text: str, *, max_chars: int) -> str:
    """Strip risky phrases and cap size; does not interpret instructions."""
    cleaned = str(text or "")
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[filtered]", cleaned)
    cleaned = cleaned.replace("\x00", "")
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3] + "..."
    return cleaned