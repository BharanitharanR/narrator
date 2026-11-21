"""Read analyser artifacts and orchestrate narration planning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checks import apply_checks, load_speech_segments, speech_windows
from .decision import build_speakers, decide, recommendation_for
from .models import NarrationPlan

REQUEST_FILENAME = "narration-request.json"
RAW_RESPONSE_FILENAME = "narration-response.raw.json"
_UNKNOWN = {"primary": "unknown", "confidence": 0.0, "alternatives": [], "reason": "No video_type classification in source analysis"}


def load_analysis(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "video" not in data:
        raise ValueError("Expected video-analysis output")
    return data


def _dims(analysis: dict[str, Any]) -> dict[str, Any]:
    return (analysis.get("video") or {}).get("dimensions") or {}


def classification_from_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """video.dimensions.video_type, else video.dimensions.semantic.video_type, else unknown."""
    dims = _dims(analysis)
    found = dims.get("video_type") or (dims.get("semantic") or {}).get("video_type")
    if not isinstance(found, dict) or not found.get("primary"):
        return dict(_UNKNOWN)
    return found


def build_planning_context(analysis: dict[str, Any]) -> dict[str, Any]:
    dims = _dims(analysis)
    sem = dict(dims.get("semantic") or {})
    tech = dims.get("technical") or {}
    screen_actions = sem.pop("screen_actions", None)
    sem.pop("video_type", None)  # given once, at the top level
    context = {
        "duration_seconds": float(tech.get("duration_seconds") or sem.get("duration_seconds") or 0),
        "video_type": classification_from_analysis(analysis),
        "semantic": sem,
        "scenes": dims.get("scenes", {}),
        "screen": dims.get("screen", {}),
    }
    if screen_actions:
        context["screen_actions"] = screen_actions
    return context


def decide_for_analysis(analysis_path: str | Path, *, mode: str | None = None, force: bool = False) -> dict[str, Any]:
    context = build_planning_context(load_analysis(analysis_path))
    result = decide(context["video_type"], mode=mode, force=force)
    return {**result, "video_type": context["video_type"].get("primary", "unknown"),
            "confidence": context["video_type"].get("confidence", 0.0), "mode": mode or "single"}


def _write_exchange(provider: Any, output_dir: Path | None) -> None:
    exchange = getattr(provider, "last_exchange", None)
    if output_dir is None or not exchange:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if exchange.get("request") is not None:
        (output_dir / REQUEST_FILENAME).write_text(json.dumps(exchange["request"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if exchange.get("raw_text") is not None:
        (output_dir / RAW_RESPONSE_FILENAME).write_text(exchange["raw_text"], encoding="utf-8")


def plan_with_provider(
    analysis_path: str | Path,
    provider: Any,
    *,
    mode: str | None = None,
    force: bool = False,
    speakers: str | None = None,
    voices: str | list[str] | None = None,
    speech_segments: str | Path | list[dict[str, float]] | None = None,
    output_path: str | Path | None = None,
    **preferences: Any,
) -> NarrationPlan:
    """Decide, then (only when narrating) call provider.create_plan, then check the result.

    When output_path is given, the provider's request and raw response are saved next to it.
    """
    context = build_planning_context(load_analysis(analysis_path))
    duration = context["duration_seconds"]
    if duration <= 0:
        raise ValueError("No valid source duration")
    video_type = context["video_type"]
    kind = video_type.get("primary", "unknown")
    decision = decide(video_type, mode=mode, force=force)
    effective_mode = mode or "single"
    speaker_list = build_speakers(effective_mode, speakers, voices)

    notes: list[str] = []
    windows: list[dict[str, Any]] = []
    if speech_segments is not None:
        segs = load_speech_segments(speech_segments) if isinstance(speech_segments, (str, Path)) else speech_segments
        windows, notes = speech_windows(segs, duration)
        context["original_speech_windows"] = [{"start_seconds": w["start_seconds"], "end_seconds": w["end_seconds"]} for w in windows]

    base = {
        "schema_version": "1.1",
        "video_type": kind,
        "classification_confidence": video_type.get("confidence", 0),
        "mode": effective_mode,
        "speakers": speaker_list,
        "decision": decision,
        "duration_seconds": duration,
        "source_analysis": str(analysis_path),
    }
    for key in ("language", "audience"):
        if preferences.get(key):
            base[key] = preferences[key]
    if preferences.get("target_words_per_minute"):
        base["target_words_per_minute"] = preferences["target_words_per_minute"]

    if not decision["narrate"]:
        raw: dict[str, Any] = {**base, "recommendation": recommendation_for(video_type, decision),
                               "segments": [], "no_voice_windows": windows, "review_notes": notes}
        return NarrationPlan.model_validate(raw)

    prefs = {**preferences, "mode": effective_mode, "speakers": speaker_list}
    out_dir = Path(output_path).resolve().parent if output_path else None
    try:
        raw = provider.create_plan(context=context, preferences=prefs)
    finally:
        _write_exchange(provider, out_dir)

    # Code owns these fields; the model's values are not trusted.
    raw.update(base)
    raw.setdefault("recommendation", recommendation_for(video_type, decision))
    model_windows = [w for w in raw.get("no_voice_windows") or [] if w.get("reason") != "original_speech" or not windows]
    raw["no_voice_windows"] = sorted(model_windows + windows, key=lambda w: (w.get("start_seconds", 0), w.get("end_seconds", 0)))
    raw["review_notes"] = list(raw.get("review_notes") or []) + notes
    apply_checks(raw)
    plan = NarrationPlan.model_validate(raw)
    if abs(plan.duration_seconds - duration) > 0.25:
        raise ValueError("Narration duration must match source video")
    return plan


def write_plan(plan: NarrationPlan, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
