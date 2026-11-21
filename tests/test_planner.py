import json
from pathlib import Path

import pytest
from conftest import analysis_dict

from narrator.models import NarrationPlan
from narrator.planner import (
    build_planning_context,
    classification_from_analysis,
    plan_with_provider,
    write_plan,
)
from narrator.prompt import build_user_prompt


def single_seg(start, end, text, pace="normal"):
    return {"start_seconds": start, "end_seconds": end, "visual_action": "User signs in",
            "narration": {"text": text, "purpose": "explain_action", "pace": pace},
            "evidence": [{"source": "action", "timestamp_seconds": start, "detail": "click"}], "confidence": 0.9}


def test_reads_type_top_level():
    a = analysis_dict()
    assert classification_from_analysis(a)["primary"] == "demo"
    assert build_planning_context(a)["duration_seconds"] == 20


def test_reads_type_from_semantic_fallback():
    a = analysis_dict(primary="tutorial", where="semantic")
    assert classification_from_analysis(a)["primary"] == "tutorial"
    assert "video_type" not in build_planning_context(a)["semantic"]


def test_top_level_wins_over_semantic():
    a = analysis_dict(where="both")
    a["video"]["dimensions"]["semantic"]["video_type"] = {"primary": "vlog", "confidence": 0.9}
    assert classification_from_analysis(a)["primary"] == "demo"


def test_legacy_is_unknown():
    vt = classification_from_analysis({"video": {"dimensions": {}}})
    assert vt["primary"] == "unknown" and vt["confidence"] == 0


def test_screen_actions_in_context():
    acts = [{"timestamp_seconds": 2, "action": "click", "target": "Sign in", "visible_text": "Sign in", "result": "dashboard"}]
    ctx = build_planning_context(analysis_dict(screen_actions=acts))
    assert ctx["screen_actions"] == acts
    assert "screen_actions" not in ctx["semantic"]
    assert "screen_actions" not in build_planning_context(analysis_dict())


def test_out_of_bounds_segment_rejected():
    with pytest.raises(ValueError):
        NarrationPlan.model_validate({"schema_version": "1.0", "video_type": "demo", "classification_confidence": 0.9, "recommendation": "recommended", "duration_seconds": 10, "segments": [single_seg(8, 12, "The page changes.")]})


def test_notes_reach_prompt_but_not_preferences(write_analysis, fake_provider):
    prov = fake_provider({"recommendation": "recommended", "segments": [single_seg(1, 5, "Sign in to Acme.")]})
    plan_with_provider(write_analysis(), prov, notes="Product is called Acme Desk; audience is support agents")
    prefs = prov.calls[0]["preferences"]
    assert prefs["notes"].startswith("Product is called Acme Desk")
    prompt = build_user_prompt(prov.calls[0]["context"], prefs)
    assert "PERSON'S NOTES" in prompt and "Acme Desk" in prompt
    assert '"notes"' not in prompt.split("PERSON'S NOTES")[0]


def test_no_notes_section_when_empty():
    assert "PERSON'S NOTES" not in build_user_prompt({}, {"notes": "  "})


def test_code_owned_fields_override_model(write_analysis, fake_provider):
    prov = fake_provider({"recommendation": "recommended", "video_type": "vlog", "mode": "dialogue",
                          "duration_seconds": 99, "segments": [single_seg(1, 5, "Sign in.")]})
    plan = plan_with_provider(write_analysis(), prov)
    assert plan.video_type == "demo" and plan.mode == "single" and plan.duration_seconds == 20
    assert plan.schema_version == "1.1" and plan.speakers[0].voice == "Kore"
    assert plan.decision.narrate is True


def test_request_and_raw_saved_next_to_output(write_analysis, fake_provider, tmp_path):
    out = tmp_path / "out" / "narration-plan.json"
    prov = fake_provider({"recommendation": "recommended", "segments": [single_seg(1, 5, "Sign in.")]})
    plan = plan_with_provider(write_analysis(), prov, output_path=out)
    write_plan(plan, out)
    assert json.loads((out.parent / "narration-request.json").read_text()) == {"contents": "prompt"}
    assert json.loads((out.parent / "narration-response.raw.json").read_text())["segments"]


def test_raw_saved_even_when_validation_fails(write_analysis, fake_provider, tmp_path):
    out = tmp_path / "narration-plan.json"
    prov = fake_provider({"recommendation": "recommended", "segments": [single_seg(1, 50, "Too far.")]})
    with pytest.raises(ValueError):
        plan_with_provider(write_analysis(), prov, output_path=out)
    assert (tmp_path / "narration-response.raw.json").exists()


def test_v10_example_still_validates():
    data = json.loads(Path(__file__).resolve().parents[1].joinpath("examples/narration-plan.json").read_text())
    assert NarrationPlan.model_validate(data).schema_version == "1.0"
