"""NarrationPlan 1.0 contract."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

VideoType = Literal["demo","tutorial","walkthrough","presentation","documentation","screen_recording","vlog","conversation","gameplay","review","advertisement","real_world","music_video","animation","news","meeting","mixed","unknown"]

class EvidenceRef(StrictModel):
    source: Literal["semantic_timeline","event","moment","edit_point","screen_text","action","manual"]
    timestamp_seconds: float = Field(ge=0)
    detail: str = Field(min_length=1,max_length=300)

class NarrationText(StrictModel):
    text: str = Field(min_length=1,max_length=500)
    purpose: Literal["introduce","explain_action","explain_result","transition","summarize","warn","context"]
    tone: Literal["neutral","friendly","confident","energetic","calm","formal"] = "neutral"
    pace: Literal["slow","normal","brisk"] = "normal"

class NarrationSegment(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    visual_action: str = Field(min_length=1,max_length=300)
    narration: NarrationText | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list,max_length=12)
    confidence: float = Field(ge=0,le=1)
    estimated_speech_seconds: float = Field(default=0,ge=0)
    fit_status: Literal["fits","tight","too_long","no_voice","needs_review"] = "fits"
    review_reason: str | None = Field(default=None,max_length=300)

    @model_validator(mode="after")
    def interval_valid(self):
        if self.end_seconds <= self.start_seconds: raise ValueError("segment end must exceed start")
        if self.narration is None and self.fit_status not in {"no_voice","needs_review"}: raise ValueError("Missing narration must be no_voice or needs_review")
        return self

class NoVoiceWindow(StrictModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    reason: Literal["original_speech","important_visual","reading_time","transition","ambient_audio","no_narration_needed","unknown"]
    detail: str = Field(default="",max_length=300)

    @model_validator(mode="after")
    def interval_valid(self):
        if self.end_seconds <= self.start_seconds: raise ValueError("window end must exceed start")
        return self

class NarrationPlan(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    video_type: VideoType
    classification_confidence: float = Field(ge=0,le=1)
    title: str = Field(default="Untitled narration",max_length=160)
    recommendation: Literal["recommended","optional","not_recommended","needs_review"]
    language: str = Field(default="en",min_length=2,max_length=32)
    audience: str = Field(default="general",max_length=120)
    overall_tone: Literal["neutral","friendly","confident","energetic","calm","formal"] = "friendly"
    target_words_per_minute: int = Field(default=145,ge=80,le=240)
    duration_seconds: float = Field(gt=0)
    segments: list[NarrationSegment] = Field(default_factory=list,max_length=1000)
    no_voice_windows: list[NoVoiceWindow] = Field(default_factory=list,max_length=500)
    review_notes: list[str] = Field(default_factory=list,max_length=100)
    source_analysis: str = "analysis.json"

    @model_validator(mode="after")
    def validate_timeline(self):
        last_start=-1.0
        for seg in self.segments:
            if seg.end_seconds > self.duration_seconds+0.25: raise ValueError("segment exceeds video duration")
            if seg.start_seconds < last_start: raise ValueError("segments must be ordered")
            last_start=seg.start_seconds
        for win in self.no_voice_windows:
            if win.end_seconds > self.duration_seconds+0.25: raise ValueError("no-voice window exceeds video duration")
        return self
