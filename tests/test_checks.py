import json

import pytest

from narrator.checks import (
    TURN_GAP_SECONDS,
    apply_checks,
    fit_status,
    load_speech_segments,
    speech_seconds,
    speech_windows,
    word_count,
)
from narrator.planner import plan_with_provider


def seg(start, end, text, pace="normal"):
    return {"start_seconds": start, "end_seconds": end, "visual_action": "x",
            "narration": {"text": text, "purpose": "explain_action", "pace": pace},
            "evidence": [{"source": "action", "timestamp_seconds": start, "detail": "x"}], "confidence": 0.9}


def words(n):
    return " ".join(["word"] * n)


def test_word_count_and_pace():
    assert word_count("Click Sign-in, then don't wait.") == 5
    assert speech_seconds(words(145), 145) == pytest.approx(60)
    assert speech_seconds(words(145), 145, "slow") == pytest.approx(60 / 0.85)
    assert speech_seconds(words(145), 145, "brisk") == pytest.approx(60 / 1.15)


def test_fit_thresholds():
    assert fit_status(9.0, 10) == "fits"
    assert fit_status(9.5, 10) == "tight"
    assert fit_status(10.0, 10) == "tight"
    assert fit_status(10.1, 10) == "too_long"


def test_recompute_overrides_model_and_flags_too_long():
    raw = {"target_words_per_minute": 120, "segments": [
        seg(0, 10, words(10)),  # 5 s
        seg(10, 12, words(10)),  # 5 s in 2 s window
    ]}
    raw["segments"][0]["estimated_speech_seconds"] = 99
    raw["segments"][0]["fit_status"] = "too_long"
    apply_checks(raw)
    a, b = raw["segments"]
    assert a["estimated_speech_seconds"] == 5.0 and a["fit_status"] == "fits"
    assert b["fit_status"] == "too_long" and "exceeds window" in b["review_reason"]
    assert b["end_seconds"] == 12  # not clamped
    assert any("segment 1" in n for n in raw["review_notes"])
    assert any("recomputed fits" in n for n in raw["review_notes"])


def test_overlap_and_ordering_flagged():
    raw = {"segments": [seg(5, 9, "b"), seg(0, 6, "a")]}
    apply_checks(raw)
    assert [s["start_seconds"] for s in raw["segments"]] == [0, 5]
    assert "overlaps segment 0" in raw["segments"][1]["review_reason"]
    assert any("reordered" in n for n in raw["review_notes"])


def test_dialogue_sum_plus_gaps():
    s = {"start_seconds": 0, "end_seconds": 20, "visual_action": "x", "confidence": 0.9, "turns": [
        {"speaker": "host", "text": words(12)}, {"speaker": "guide", "text": words(12), "pace": "slow"}]}
    raw = {"target_words_per_minute": 120, "segments": [s]}
    apply_checks(raw)
    expected = 6 + 6 / 0.85 + TURN_GAP_SECONDS
    assert s["estimated_speech_seconds"] == pytest.approx(expected, abs=0.01)


def test_silent_segment_marked_no_voice():
    raw = {"segments": [{"start_seconds": 0, "end_seconds": 2, "visual_action": "x", "confidence": 0.5, "fit_status": "fits"}]}
    apply_checks(raw)
    assert raw["segments"][0]["fit_status"] == "no_voice"


def test_speech_file_formats(tmp_path):
    p = tmp_path / "speech.json"
    p.write_text(json.dumps([{"start_seconds": 1, "end_seconds": 2.5}]))
    assert load_speech_segments(p) == [{"start_seconds": 1.0, "end_seconds": 2.5}]
    p.write_text(json.dumps({"segments": [{"start_seconds": 3, "end_seconds": 4}]}))
    assert load_speech_segments(p)[0]["start_seconds"] == 3.0


def test_speech_windows_report_cut():
    windows, notes = speech_windows([{"start_seconds": 18, "end_seconds": 25}, {"start_seconds": 30, "end_seconds": 31}], 20)
    assert windows == [{"start_seconds": 18, "end_seconds": 20, "reason": "original_speech", "detail": "original speech detected in the source audio"}]
    assert len(notes) == 2


def test_speech_segments_become_windows_and_flag_overlap(write_analysis, fake_provider, tmp_path):
    speech = tmp_path / "speech.json"
    speech.write_text(json.dumps([{"start_seconds": 6, "end_seconds": 9}]))
    prov = fake_provider({"recommendation": "recommended",
                          "segments": [seg(0, 5, "Sign in."), seg(8, 12, "Open the report.")],
                          "no_voice_windows": [{"start_seconds": 14, "end_seconds": 16, "reason": "reading_time"}]})
    plan = plan_with_provider(write_analysis(), prov, speech_segments=speech)
    ctx = prov.calls[0]["context"]
    assert ctx["original_speech_windows"] == [{"start_seconds": 6, "end_seconds": 9}]
    reasons = [(w.start_seconds, w.reason) for w in plan.no_voice_windows]
    assert reasons == [(6, "original_speech"), (14, "reading_time")]
    assert plan.segments[0].review_reason is None
    assert "overlaps original speech" in plan.segments[1].review_reason
    assert any("original speech" in n for n in plan.review_notes)
