"""NarrationPlan contract (1.1, backward compatible with 1.0)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


VideoType = Literal[
    "demo", "tutorial", "walkthrough", "presentation", "documentation", "screen_recording",
    "vlog", "conversation", "gameplay", "review", "advertisement", "real_world", "music_video",
    "animation", "news", "meeting", "mixed", "unknown",
]
Tone = Literal["neutral", "friendly", "confident", "energetic", "calm", "formal"]
Pace = Literal["slow", "normal", "brisk"]
Mode = Literal["single", "dialogue"]


class EvidenceRef(StrictModel):
    source: Literal["semantic_timeline", "event", "moment", "edit_point", "screen_text", "action", "manual"]
    timestamp_seconds: float = Field(ge=0)
    detail: str = Field(min_length=1, max_length=300)


class NarrationText(StrictModel):
    text: str = Field(min_length=1, max_length=500)
    purpose: Literal["introduce", "explain_action", "explain_result", "transition", "summarize", "warn", "context"]
    tone: Tone = "neutral"
    pace: Pace = "normal"


class DialogueTurn(StrictModel):
    speaker: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=500)
    style: str = Field(default="", max_length=200)
    tone: Tone = "neutral"
    pace: Pace = "normal"


class Speaker(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=80)
    role: str = Field(default="", max_length=120)
    voice: str = Field(min_length=1, max_length=60)


class Decision(StrictModel):
    narrate: bool
    reason: str = Field(min_length=1, max_length=400)


class NarrationSegment(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    visual_action: str = Field(min_length=1, max_length=300)
    narration: NarrationText | None = None
    turns: list[DialogueTurn] | None = Field(default=None, max_length=12)
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    confidence: float = Field(ge=0, le=1)
    estimated_speech_seconds: float = Field(default=0, ge=0)
    fit_status: Literal["fits", "tight", "too_long", "no_voice", "needs_review"] = "fits"
    review_reason: str | None = Field(default=None, max_length=600)

    @property
    def has_speech(self) -> bool:
        return self.narration is not None or bool(self.turns)

    @model_validator(mode="after")
    def interval_valid(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("segment end must exceed start")
        if self.narration is not None and self.turns:
            raise ValueError("segment has both narration and turns")
        if not self.has_speech and self.fit_status not in {"no_voice", "needs_review"}:
            raise ValueError("Missing narration must be no_voice or needs_review")
        return self


class NoVoiceWindow(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    reason: Literal["original_speech", "important_visual", "reading_time", "transition", "ambient_audio", "no_narration_needed", "unknown"]
    detail: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def interval_valid(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("window end must exceed start")
        return self


class NarrationPlan(StrictModel):
    schema_version: Literal["1.0", "1.1"] = "1.1"
    video_type: VideoType
    classification_confidence: float = Field(ge=0, le=1)
    title: str = Field(default="Untitled narration", max_length=160)
    recommendation: Literal["recommended", "optional", "not_recommended", "needs_review"]
    mode: Mode = "single"
    speakers: list[Speaker] = Field(default_factory=list, max_length=2)
    decision: Decision | None = None
    language: str = Field(default="en", min_length=2, max_length=32)
    audience: str = Field(default="general", max_length=120)
    overall_tone: Tone = "friendly"
    target_words_per_minute: int = Field(default=145, ge=80, le=240)
    duration_seconds: float = Field(gt=0)
    segments: list[NarrationSegment] = Field(default_factory=list, max_length=1000)
    no_voice_windows: list[NoVoiceWindow] = Field(default_factory=list, max_length=500)
    review_notes: list[str] = Field(default_factory=list, max_length=200)
    source_analysis: str = "analysis.json"

    @model_validator(mode="after")
    def validate_timeline(self):
        last_start = -1.0
        for seg in self.segments:
            if seg.end_seconds > self.duration_seconds + 0.25:
                raise ValueError("segment exceeds video duration")
            if seg.start_seconds < last_start:
                raise ValueError("segments must be ordered")
            last_start = seg.start_seconds
        for win in self.no_voice_windows:
            if win.end_seconds > self.duration_seconds + 0.25:
                raise ValueError("no-voice window exceeds video duration")
        if self.schema_version == "1.0":
            return self
        expected = 2 if self.mode == "dialogue" else 1
        if len(self.speakers) != expected:
            raise ValueError(f"mode {self.mode} needs {expected} speaker(s)")
        ids = [s.id for s in self.speakers]
        if len(set(ids)) != len(ids):
            raise ValueError("speaker ids must be unique")
        for seg in self.segments:
            if self.mode == "single" and seg.turns:
                raise ValueError("single mode segments use narration, not turns")
            if self.mode == "dialogue" and seg.narration is not None:
                raise ValueError("dialogue mode segments use turns, not narration")
            for turn in seg.turns or []:
                if turn.speaker not in ids:
                    raise ValueError(f"turn speaker {turn.speaker!r} is not in speakers")
        if self.decision is not None and not self.decision.narrate and any(s.has_speech for s in self.segments):
            raise ValueError("decision.narrate is false but segments contain speech")
        return self
