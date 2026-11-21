"""Narrate-or-not rule and speaker defaults, decided in code without a model call."""
from __future__ import annotations

from typing import Any

NARRATABLE_TYPES = frozenset(
    {"demo", "tutorial", "walkthrough", "presentation", "documentation", "screen_recording", "review"}
)
MIN_CONFIDENCE = 0.6

DEFAULT_SINGLE_VOICE = "Kore"
DEFAULT_DIALOGUE_VOICES = ("Puck", "Kore")


def decide(video_type: dict[str, Any], *, mode: str | None = None, force: bool = False) -> dict[str, Any]:
    """Return {narrate, reason}. An explicit mode or force overrides a "no"."""
    kind = str(video_type.get("primary") or "unknown")
    try:
        conf = float(video_type.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if kind in NARRATABLE_TYPES and conf >= MIN_CONFIDENCE:
        return {"narrate": True, "reason": f"video type {kind} (confidence {conf:.2f}) benefits from narration"}
    if kind in NARRATABLE_TYPES:
        base = f"video type {kind} has low confidence {conf:.2f} (< {MIN_CONFIDENCE})"
    elif kind == "unknown":
        base = "video type is unknown"
    else:
        base = f"video type {kind} (confidence {conf:.2f}) is not one that narration is planned for"
    if force or mode:
        how = "--force" if force else f"--mode {mode}"
        return {"narrate": True, "reason": f"{base}; narrating because {how} was given"}
    return {"narrate": False, "reason": f"{base}; not narrating"}


def recommendation_for(video_type: dict[str, Any], decision: dict[str, Any]) -> str:
    kind = video_type.get("primary") or "unknown"
    conf = float(video_type.get("confidence") or 0.0)
    if decision["narrate"]:
        return "recommended" if kind in NARRATABLE_TYPES and conf >= MIN_CONFIDENCE else "optional"
    return "needs_review" if kind in NARRATABLE_TYPES or kind == "unknown" else "not_recommended"


def build_speakers(mode: str, spec: str | None = None, voices: str | list[str] | None = None) -> list[dict[str, str]]:
    """Speakers for a mode. spec is "Name:role,Name:role"; voices is "Puck,Kore".

    Ids are fixed by position: "narrator" in single mode, "host" and "guide" in dialogue.
    """
    if mode == "dialogue":
        ids = ["host", "guide"]
        defaults = [("Host", "curious host"), ("Guide", "guide who explains what is on screen")]
        default_voices = list(DEFAULT_DIALOGUE_VOICES)
    elif mode == "single":
        ids = ["narrator"]
        defaults = [("Narrator", "narrator")]
        default_voices = [DEFAULT_SINGLE_VOICE]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    pairs = defaults
    if spec:
        parsed = []
        for item in spec.split(","):
            if not item.strip():
                continue
            name, _, role = item.partition(":")
            parsed.append((name.strip(), role.strip()))
        if len(parsed) != len(ids):
            raise ValueError(f"--speakers needs {len(ids)} entries for {mode} mode, got {len(parsed)}")
        pairs = [(n or d[0], r or d[1]) for (n, r), d in zip(parsed, defaults)]
    if isinstance(voices, str):
        voices = [v.strip() for v in voices.split(",") if v.strip()]
    voice_list = list(voices) if voices else default_voices
    if len(voice_list) != len(ids):
        raise ValueError(f"--voices needs {len(ids)} entries for {mode} mode, got {len(voice_list)}")
    return [{"id": i, "name": n, "role": r, "voice": v} for i, (n, r), v in zip(ids, pairs, voice_list)]
