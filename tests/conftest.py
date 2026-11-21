import json

import pytest


def analysis_dict(primary="demo", confidence=0.94, where="top", duration=20.0, screen_actions=None):
    vt = {"primary": primary, "confidence": confidence, "alternatives": [], "reason": "test"}
    sem = {"timeline": [{"timestamp_seconds": 1, "description": "Login page is visible"}]}
    dims = {"technical": {"duration_seconds": duration}, "semantic": sem}
    if where in ("top", "both"):
        dims["video_type"] = vt
    if where in ("semantic", "both"):
        sem["video_type"] = vt
    if screen_actions is not None:
        sem["screen_actions"] = screen_actions
    return {"video": {"dimensions": dims}}


@pytest.fixture
def write_analysis(tmp_path):
    def _write(**kw):
        path = tmp_path / "analysis.json"
        path.write_text(json.dumps(analysis_dict(**kw)))
        return path
    return _write


class FakeProvider:
    """Returns a canned plan; records calls; mimics last_exchange."""

    def __init__(self, response):
        self.response = response
        self.calls = []
        self.last_exchange = None

    def create_plan(self, *, context, preferences):
        self.calls.append({"context": context, "preferences": preferences})
        self.last_exchange = {"request": {"contents": "prompt"}, "raw_text": json.dumps(self.response), "model_used": "fake"}
        return json.loads(json.dumps(self.response))


@pytest.fixture
def fake_provider():
    return FakeProvider
