"""Deterministic checks applied to a plan after the model returns it.

Nothing is clamped or dropped: problems are flagged in review_reason and review_notes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PACE_FACTOR = {"slow": 0.85, "normal": 1.0, "brisk": 1.15}
TURN_GAP_SECONDS = 0.3
TIGHT_RATIO = 0.9
_WORD = re.compile(r"[\w']+(?:[-.][\w']+)*", re.UNICODE)


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def speech_seconds(text: str, wpm: int, pace: str = "normal") -> float:
    rate = wpm * PACE_FACTOR.get(pace, 1.0)
    return word_count(text) / rate * 60.0


def segment_speech_seconds(seg: dict[str, Any], wpm: int) -> float:
    turns = seg.get("turns") or []
    if turns:
        total = sum(speech_seconds(t.get("text", ""), wpm, t.get("pace", "normal")) for t in turns)
        return total + TURN_GAP_SECONDS * (len(turns) - 1)
    narration = seg.get("narration")
    if narration:
        return speech_seconds(narration.get("text", ""), wpm, narration.get("pace", "normal"))
    return 0.0


def fit_status(estimate: float, window: float) -> str:
    if estimate <= window * TIGHT_RATIO:
        return "fits"
    if estimate <= window:
        return "tight"
    return "too_long"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _flag(seg: dict[str, Any], reason: str) -> None:
    existing = seg.get("review_reason")
    seg["review_reason"] = f"{existing}; {reason}" if existing else reason


def apply_checks(raw: dict[str, Any]) -> dict[str, Any]:
    """Recompute speech estimates and fit, flag overlaps. Mutates and returns raw."""
    wpm = int(raw.get("target_words_per_minute") or 145)
    notes: list[str] = list(raw.get("review_notes") or [])
    segments: list[dict[str, Any]] = list(raw.get("segments") or [])
    ordered = sorted(segments, key=lambda s: float(s.get("start_seconds", 0)))
    if ordered != segments:
        notes.append("segments were not in start order; reordered by start time")
        segments = ordered
    speech_windows = [w for w in raw.get("no_voice_windows") or [] if w.get("reason") == "original_speech"]
    for index, seg in enumerate(segments):
        start, end = float(seg.get("start_seconds", 0)), float(seg.get("end_seconds", 0))
        window = end - start
        has_speech = bool(seg.get("narration")) or bool(seg.get("turns"))
        if not has_speech:
            seg["estimated_speech_seconds"] = 0.0
            if seg.get("fit_status") not in {"no_voice", "needs_review"}:
                seg["fit_status"] = "no_voice"
            continue
        estimate = round(segment_speech_seconds(seg, wpm), 2)
        reported = seg.get("estimated_speech_seconds")
        seg["estimated_speech_seconds"] = estimate
        status = fit_status(estimate, window) if window > 0 else "too_long"
        if reported is not None and seg.get("fit_status") not in (None, status) and seg.get("fit_status") != "needs_review":
            notes.append(f"segment {index}: model said {seg.get('fit_status')}, recomputed {status}")
        if seg.get("fit_status") != "needs_review":
            seg["fit_status"] = status
        if status == "too_long":
            reason = f"speech {estimate:.1f}s exceeds window {window:.1f}s"
            _flag(seg, reason)
            notes.append(f"segment {index} ({start:.1f}-{end:.1f}s): {reason}")
        for w in speech_windows:
            ov = _overlap(start, end, float(w["start_seconds"]), float(w["end_seconds"]))
            if ov > 0:
                reason = f"overlaps original speech {w['start_seconds']:.1f}-{w['end_seconds']:.1f}s by {ov:.1f}s"
                _flag(seg, reason)
                notes.append(f"segment {index} ({start:.1f}-{end:.1f}s): {reason}")
    for index in range(1, len(segments)):
        prev, seg = segments[index - 1], segments[index]
        ov = float(prev.get("end_seconds", 0)) - float(seg.get("start_seconds", 0))
        if ov > 0:
            reason = f"overlaps segment {index - 1} by {ov:.2f}s"
            _flag(seg, reason)
            notes.append(f"segment {index}: {reason}")
    raw["segments"] = segments
    raw["review_notes"] = notes
    return raw


def load_speech_segments(path: str | Path) -> list[dict[str, float]]:
    """Read speech.json: a list of {start_seconds, end_seconds}, or {"segments": [...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("segments") or data.get("speech_segments") or []
    if not isinstance(data, list):
        raise ValueError("speech segments must be a list of {start_seconds, end_seconds}")
    out = []
    for item in data:
        out.append({"start_seconds": float(item["start_seconds"]), "end_seconds": float(item["end_seconds"])})
    return out


def speech_windows(segments: list[dict[str, float]], duration: float) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn speech segments into original_speech no-voice windows; report any cut at the video end."""
    windows: list[dict[str, Any]] = []
    notes: list[str] = []
    for s in sorted(segments, key=lambda x: x["start_seconds"]):
        start, end = max(0.0, s["start_seconds"]), s["end_seconds"]
        if end > duration:
            notes.append(f"speech segment {start:.2f}-{end:.2f}s ends after the video ({duration:.2f}s); window cut at the video end")
            end = duration
        if end - start <= 0:
            notes.append(f"speech segment {s['start_seconds']:.2f}-{s['end_seconds']:.2f}s is empty or outside the video; ignored")
            continue
        windows.append({"start_seconds": round(start, 3), "end_seconds": round(end, 3), "reason": "original_speech", "detail": "original speech detected in the source audio"})
    return windows, notes
