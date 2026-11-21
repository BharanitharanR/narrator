import json
import runpy
from pathlib import Path

import pytest

from narrator.checks import TURN_GAP_SECONDS
from narrator.models import NarrationPlan
from narrator.planner import plan_with_provider
from narrator.prompt import build_user_prompt, system_prompt

ROOT = Path(__file__).resolve().parents[1]


def dialogue_seg(start, end, turns):
    return {"start_seconds": start, "end_seconds": end, "visual_action": "x", "confidence": 0.9,
            "evidence": [{"source": "action", "timestamp_seconds": start, "detail": "x"}],
            "turns": [{"speaker": s, "text": t} for s, t in turns]}


def test_dialogue_plan(write_analysis, fake_provider):
    prov = fake_provider({"recommendation": "recommended", "segments": [
        dialogue_seg(0, 10, [("host", "What happens first?"), ("guide", "The login page loads.")])]})
    plan = plan_with_provider(write_analysis(), prov, mode="dialogue", speakers="Maya:host,Arun:guide")
    assert plan.mode == "dialogue" and [s.name for s in plan.speakers] == ["Maya", "Arun"]
    prefs = prov.calls[0]["preferences"]
    assert prefs["mode"] == "dialogue" and prefs["speakers"][0]["id"] == "host"
    seg = plan.segments[0]
    assert seg.estimated_speech_seconds == pytest.approx((3 + 4) / 145 * 60 + TURN_GAP_SECONDS, abs=0.01)


def test_unknown_speaker_rejected(write_analysis, fake_provider):
    prov = fake_provider({"recommendation": "recommended", "segments": [dialogue_seg(0, 10, [("narrator", "Hi there.")])]})
    with pytest.raises(ValueError, match="not in speakers"):
        plan_with_provider(write_analysis(), prov, mode="dialogue")


def test_mode_shape_enforced():
    base = {"video_type": "demo", "classification_confidence": 0.9, "recommendation": "recommended", "duration_seconds": 20}
    with pytest.raises(ValueError, match="needs 2"):
        NarrationPlan.model_validate({**base, "mode": "dialogue", "speakers": [{"id": "a", "name": "A", "voice": "Kore"}]})
    with pytest.raises(ValueError, match="use narration"):
        NarrationPlan.model_validate({**base, "speakers": [{"id": "narrator", "name": "N", "voice": "Kore"}],
                                      "segments": [dialogue_seg(0, 5, [("narrator", "Hi.")])]})
    with pytest.raises(ValueError, match="both"):
        NarrationPlan.model_validate({**base, "schema_version": "1.0", "segments": [
            {**dialogue_seg(0, 5, [("a", "Hi.")]), "narration": {"text": "Hi.", "purpose": "introduce"}}]})


def test_dialogue_prompt_rules():
    p = system_prompt("dialogue")
    assert "no repeating" in p and "fewer moments" in p and "No filler banter" in p
    assert "MODE single" in system_prompt("single")
    assert "dialogue mode" in build_user_prompt({}, {"mode": "dialogue"})


@pytest.mark.parametrize("name", ["narration-plan.json", "narration-plan.dialogue.json"])
def test_examples_validate(name):
    NarrationPlan.model_validate(json.loads((ROOT / "examples" / name).read_text()))


def test_schema_file_is_current():
    gen = runpy.run_path(str(ROOT / "scripts" / "generate_schema.py"))
    committed = json.loads((ROOT / "references" / "narration-plan.schema.json").read_text())
    assert committed == gen["schema"](), "run scripts/generate_schema.py"
