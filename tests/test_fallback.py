import json

import pytest

from narrator.fallback import (
    DEFAULT_MODELS,
    is_retryable_model_error,
    model_chain,
    run_with_model_fallback,
)
from narrator.gemini import GeminiNarrationProvider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("NARRATOR_MODELS", "NARRATOR_MAX_MODEL_ATTEMPTS", "NARRATOR_MODEL_FALLBACK"):
        monkeypatch.delenv(k, raising=False)


def test_default_chain_capped_at_three():
    assert model_chain() == list(DEFAULT_MODELS[:3])
    assert model_chain("gemini-2.5-flash") == ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-4-hydrogen"]


def test_env_models_and_limit(monkeypatch):
    monkeypatch.setenv("NARRATOR_MODELS", "a, b ,c,d")
    monkeypatch.setenv("NARRATOR_MAX_MODEL_ATTEMPTS", "2")
    assert model_chain() == ["a", "b"]
    assert model_chain("b") == ["b", "a"]


def test_fallback_disabled(monkeypatch):
    monkeypatch.setenv("NARRATOR_MODEL_FALLBACK", "0")
    assert model_chain("x") == ["x"]
    assert model_chain() == ["gemini-3.8-flash"]


class Coded(Exception):
    def __init__(self, code):
        super().__init__(f"error {code}")
        self.code = code


def test_retryable_classification():
    assert is_retryable_model_error(Coded(429))
    assert is_retryable_model_error(RuntimeError("503 UNAVAILABLE"))
    assert is_retryable_model_error(RuntimeError("RESOURCE_EXHAUSTED quota"))
    assert not is_retryable_model_error(ValueError("RESOURCE_EXHAUSTED"))
    assert not is_retryable_model_error(Coded(400))


def test_run_moves_on_only_for_retryable():
    logs = []
    seen = []

    def attempt(model):
        seen.append(model)
        if model == "a":
            raise Coded(429)
        return "ok"

    assert run_with_model_fallback(["a", "b", "c"], attempt, log=logs.append) == ("ok", "b")
    assert seen == ["a", "b"] and logs[-1] == "Gemini model used: b"

    def bad(model):
        raise Coded(400)

    with pytest.raises(Coded):
        run_with_model_fallback(["a", "b"], bad, log=logs.append)

    def busy(model):
        raise Coded(503)

    with pytest.raises(RuntimeError, match="Every Gemini model"):
        run_with_model_fallback(["a", "b"], busy, log=logs.append)


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return FakeResponse(out)


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def test_gemini_provider_with_fake_client(capsys):
    client = FakeClient([Coded(429), json.dumps({"segments": []})])
    prov = GeminiNarrationProvider(client=client)
    data = prov.create_plan(context={"duration_seconds": 5}, preferences={"mode": "dialogue", "notes": "N"})
    assert data == {"segments": []}
    assert [c["model"] for c in client.models.calls] == ["gemini-3.8-flash", "gemini-4-hydrogen"]
    assert prov.last_exchange["model_used"] == "gemini-4-hydrogen"
    req = prov.last_exchange["request"]
    assert "MODE dialogue" in req["system_instruction"] and "N" in req["contents"]
    assert "api_key" not in json.dumps(req).lower()
    assert "Gemini model used: gemini-4-hydrogen" in capsys.readouterr().err


def test_gemini_provider_keeps_raw_on_bad_json():
    prov = GeminiNarrationProvider(client=FakeClient(["not json"]))
    with pytest.raises(ValueError):
        prov.create_plan(context={}, preferences={})
    assert prov.last_exchange["raw_text"] == "not json"
