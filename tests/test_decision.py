import json

import pytest

from narrator.cli import build_parser, cmd_plan, main
from narrator.decision import build_speakers, decide
from narrator.planner import plan_with_provider


@pytest.mark.parametrize("kind", ["demo", "tutorial", "walkthrough", "presentation", "documentation", "screen_recording", "review"])
def test_narratable_types(kind):
    assert decide({"primary": kind, "confidence": 0.6})["narrate"] is True


def test_low_confidence_and_other_types():
    d = decide({"primary": "demo", "confidence": 0.59})
    assert d["narrate"] is False and "low confidence" in d["reason"]
    assert decide({"primary": "vlog", "confidence": 0.99})["narrate"] is False
    assert decide({"primary": "unknown", "confidence": 0})["narrate"] is False


def test_force_and_explicit_mode_override():
    assert decide({"primary": "vlog", "confidence": 0.9}, force=True)["narrate"] is True
    d = decide({"primary": "demo", "confidence": 0.2}, mode="single")
    assert d["narrate"] is True and "--mode single" in d["reason"]


class Boom:
    last_exchange = None

    def create_plan(self, **kw):
        raise AssertionError("Gemini must not be called")


def test_no_narration_skips_provider(write_analysis, tmp_path):
    plan = plan_with_provider(write_analysis(primary="vlog", confidence=0.95), Boom(), output_path=tmp_path / "p.json")
    assert plan.decision.narrate is False and plan.segments == []
    assert plan.recommendation == "not_recommended"
    assert not (tmp_path / "narration-request.json").exists()


def test_no_narration_keeps_speech_windows(write_analysis):
    plan = plan_with_provider(write_analysis(confidence=0.3), Boom(), speech_segments=[{"start_seconds": 1, "end_seconds": 2}])
    assert plan.recommendation == "needs_review"
    assert plan.no_voice_windows[0].reason == "original_speech"


def test_decide_cli(write_analysis, capsys):
    path = write_analysis(primary="vlog", confidence=0.9)
    main(["decide", str(path)])
    out = json.loads(capsys.readouterr().out)
    assert out["narrate"] is False and out["video_type"] == "vlog"
    main(["decide", str(path), "--mode", "dialogue"])
    assert json.loads(capsys.readouterr().out)["narrate"] is True


def test_plan_cli_no_narration_writes_plan_without_gemini(write_analysis, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = tmp_path / "narration-plan.json"
    main(["plan", str(write_analysis(primary="gameplay", confidence=0.9)), "-o", str(out)])
    data = json.loads(out.read_text())
    assert data["decision"]["narrate"] is False and data["segments"] == []


def test_plan_cli_with_fake_provider(write_analysis, fake_provider, tmp_path):
    out = tmp_path / "narration-plan.json"
    args = build_parser().parse_args(["plan", str(write_analysis()), "-o", str(out), "--notes", "Acme", "--voices", "Charon"])
    prov = fake_provider({"recommendation": "recommended", "segments": []})
    cmd_plan(args, provider=prov)
    data = json.loads(out.read_text())
    assert data["speakers"][0]["voice"] == "Charon"
    assert prov.calls[0]["preferences"]["notes"] == "Acme"
    assert (tmp_path / "narration-response.raw.json").exists()


def test_speakers_defaults_and_flags():
    assert [s["voice"] for s in build_speakers("dialogue")] == ["Puck", "Kore"]
    assert [s["id"] for s in build_speakers("dialogue")] == ["host", "guide"]
    sp = build_speakers("dialogue", "Maya:curious host,Arun:product guide", "Puck,Charon")
    assert sp[0]["name"] == "Maya" and sp[1]["role"] == "product guide" and sp[1]["voice"] == "Charon"
    with pytest.raises(ValueError):
        build_speakers("dialogue", "OnlyOne:host")
    with pytest.raises(ValueError):
        build_speakers("single", voices="Puck,Kore")
