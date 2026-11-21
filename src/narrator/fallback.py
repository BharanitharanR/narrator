"""Gemini model fallback: try the next model only on quota, rate-limit or server errors.

A small equivalent of video-analysis's helpers; narrator does not import video-analysis.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")

DEFAULT_MODELS = (
    "gemini-3.8-flash",
    "gemini-4-hydrogen",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
)

# Errors that mean "this model, right now", not "this request". Anything else
# (bad input, invalid JSON, validation failure) is raised at once, since another
# model would cost another paid call for the same failure.
_RETRYABLE_MARKERS = (
    "INTERNAL", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED",
    "OVERLOADED", "QUOTA", "RATE LIMIT", "TOO MANY REQUESTS",
)
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def is_retryable_model_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ValueError)):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_CODES:
        return True
    text = str(exc).upper()
    return any(marker in text for marker in _RETRYABLE_MARKERS) or any(
        f" {c}" in text or f"{c} " in text for c in ("429", "503")
    )


def model_chain(primary: str | None = None, defaults: Iterable[str] = DEFAULT_MODELS) -> list[str]:
    """The requested model first, then NARRATOR_MODELS (or defaults), without repeats.

    NARRATOR_MODEL_FALLBACK=0 disables fallback. At most NARRATOR_MAX_MODEL_ATTEMPTS
    models (default 3) are tried: each attempt is a paid call.
    """
    configured = os.environ.get("NARRATOR_MODELS")
    rest = [m.strip() for m in configured.split(",")] if configured else list(defaults)
    first = primary or (rest[0] if rest else None)
    if not first:
        raise ValueError("No Gemini model configured")
    if os.environ.get("NARRATOR_MODEL_FALLBACK", "1").strip() == "0":
        return [first]
    chain: list[str] = []
    for model in [first, *rest]:
        if model and model not in chain:
            chain.append(model)
    try:
        limit = int(os.environ.get("NARRATOR_MAX_MODEL_ATTEMPTS", "3"))
    except ValueError:
        limit = 3
    return chain[: max(1, limit)]


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_with_model_fallback(
    models: list[str],
    attempt: Callable[[str], T],
    *,
    log: Callable[[str], None] = _stderr,
) -> tuple[T, str]:
    """Call attempt(model) down the chain; return (result, model_used)."""
    failures: list[str] = []
    for index, model in enumerate(models):
        try:
            result = attempt(model)
        except Exception as exc:
            if not is_retryable_model_error(exc):
                raise
            summary = " ".join(str(exc).split())[:240]
            failures.append(f"{model}: {summary}")
            if index + 1 < len(models):
                log(f"Gemini model {model} unavailable ({summary}); trying {models[index + 1]}")
            continue
        log(f"Gemini model used: {model}")
        return result, model
    raise RuntimeError("Every Gemini model in the fallback chain failed: " + " | ".join(failures))
