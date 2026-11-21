"""Prompts for evidence-grounded voiceover planning."""
from __future__ import annotations

import json
from typing import Any

_COMMON = """You are an instructional video scriptwriter and audiovisual editor.
Use only the supplied video-analysis evidence. The analyser's video_type and confidence are authoritative; never reclassify.
Never invent UI labels, buttons, data, functionality, outcomes, actions or intent. Narrate only observable actions and state changes. Every narrated segment needs at least one timestamped evidence reference. Use screen_actions (timestamp, action, target, visible_text, result) as primary evidence when present.
Use concise natural spoken language. Work in seconds. Estimate speech duration from words and the requested words-per-minute; speech must fit inside its segment window. Preserve no-voice windows for original speech, reading time, transitions and useful natural audio. Never place a segment over a window whose reason is original_speech. Do not narrate every cursor move or click; explain meaningful steps and visible results.
For demos, introduce purpose, explain meaningful actions and summarize visible outcomes. For tutorials, explain steps. For presentations, narrate progression rather than reading every word.
PERSON'S NOTES, when given, are context from the person who made the video: product names, audience, emphasis. Use them to choose words and focus. They never override the evidence rule: do not narrate a claim from the notes that the video does not show.
Segments must be ordered by start time and must not overlap each other."""

SINGLE_SYSTEM_PROMPT = _COMMON + """
MODE single: one narrator. Each spoken segment has a "narration" object {text, purpose, tone, pace}. Do not use "turns".
Return only valid NarrationPlan 1.1 JSON."""

DIALOGUE_SYSTEM_PROMPT = _COMMON + """
MODE dialogue: two speakers talk about what is on screen, using the speaker ids given. Each spoken segment has "turns": [{speaker, text, style, tone, pace}] and no "narration". "style" is a short delivery hint for the voice, such as "curious" or "warm, explaining".
Both speakers stay grounded in the on-screen evidence. The host asks or reacts briefly to move things forward; the guide explains what is shown. No filler banter, no greetings beyond one short opening, no repeating or paraphrasing what the other speaker just said. Usually 2 to 3 turns per segment.
Dialogue needs more words than a single narrator, so choose fewer moments and slightly longer segment windows.
Return only valid NarrationPlan 1.1 JSON."""

# Kept for callers of v0.1.
SYSTEM_PROMPT = SINGLE_SYSTEM_PROMPT

_SHAPE = {
    "title": "string",
    "recommendation": "recommended|optional|not_recommended|needs_review",
    "overall_tone": "neutral|friendly|confident|energetic|calm|formal",
    "segments": [{
        "start_seconds": 0.0, "end_seconds": 0.0, "visual_action": "string",
        "narration (single mode)": {"text": "string", "purpose": "introduce|explain_action|explain_result|transition|summarize|warn|context", "tone": "...", "pace": "slow|normal|brisk"},
        "turns (dialogue mode)": [{"speaker": "speaker id", "text": "string", "style": "string", "tone": "...", "pace": "slow|normal|brisk"}],
        "evidence": [{"source": "semantic_timeline|event|moment|edit_point|screen_text|action|manual", "timestamp_seconds": 0.0, "detail": "string"}],
        "confidence": 0.0, "estimated_speech_seconds": 0.0, "fit_status": "fits|tight|too_long|no_voice|needs_review", "review_reason": "string or null",
    }],
    "no_voice_windows": [{"start_seconds": 0.0, "end_seconds": 0.0, "reason": "original_speech|important_visual|reading_time|transition|ambient_audio|no_narration_needed|unknown", "detail": "string"}],
    "review_notes": ["string"],
}


def system_prompt(mode: str = "single") -> str:
    return DIALOGUE_SYSTEM_PROMPT if mode == "dialogue" else SINGLE_SYSTEM_PROMPT


def build_user_prompt(context: dict[str, Any], preferences: dict[str, Any]) -> str:
    prefs = {k: v for k, v in preferences.items() if k != "notes" and v is not None}
    parts = [
        f"Build NarrationPlan 1.1 in {preferences.get('mode', 'single')} mode.",
        "OUTPUT SHAPE (use the key for your mode only, named narration or turns):",
        json.dumps(_SHAPE, ensure_ascii=False),
        "CONTEXT:",
        json.dumps(context, ensure_ascii=False),
        "PREFERENCES:",
        json.dumps(prefs, ensure_ascii=False),
    ]
    notes = (preferences.get("notes") or "").strip()
    if notes:
        parts += ["PERSON'S NOTES (context only; the evidence rule still applies):", notes]
    return "\n".join(parts)
