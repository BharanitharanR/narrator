import pytest
from narrator.models import NarrationPlan
from narrator.planner import classification_from_analysis,build_planning_context

def test_reads_type():
    a={"video":{"dimensions":{"technical":{"duration_seconds":20},"video_type":{"primary":"demo","confidence":0.94}}}}
    assert classification_from_analysis(a)["primary"]=="demo"
    assert build_planning_context(a)["duration_seconds"]==20

def test_legacy_is_unknown():
    assert classification_from_analysis({"video":{"dimensions":{}}})["primary"]=="unknown"

def test_out_of_bounds_segment_rejected():
    with pytest.raises(ValueError):
        NarrationPlan.model_validate({"video_type":"demo","classification_confidence":0.9,"recommendation":"recommended","duration_seconds":10,"segments":[{"start_seconds":8,"end_seconds":12,"visual_action":"screen changes","narration":{"text":"The page changes.","purpose":"explain_result"},"confidence":0.8}]})
