"""Voiceover stage tests. Fake TTS client only; no network."""
import base64
import io
import json
import shutil
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from narrator import voice as v

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
RATE = 24000
CHARS_PER_SECOND = 15.0


# ------------------------------------------------------------------ fakes

def speech(seconds: float, seed: int = 0) -> np.ndarray:
    """Speech-like audio: tone bursts with short (< 0.5 s) intra-word dips."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    sig = 0.4 * np.sin(2 * np.pi * (180 + 40 * seed % 3) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    word = int(0.35 * RATE)
    dip = int(0.08 * RATE)
    for start in range(word, n, word + dip):
        sig[start:start + dip] *= 0.05
    return (sig + 0.002 * rng.standard_normal(n)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return (0.001 * np.random.default_rng(99).standard_normal(int(seconds * RATE))).astype(np.float32)


def wav_bytes(samples: np.ndarray, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def take(texts, gap=0.9, edge=0.3) -> np.ndarray:
    parts = [silence(edge)]
    for i, text in enumerate(texts):
        parts.append(speech(max(0.4, len(text) / CHARS_PER_SECOND), seed=i))
        parts.append(silence(gap if i < len(texts) - 1 else edge))
    return np.concatenate(parts)


class FakeTTS:
    """Returns a synthetic take per request. `fail` maps model -> exception to raise.
    `merge_full` makes full-script takes lose their pauses (forces per-line fallback)."""

    def __init__(self, fail=None, merge_full=False, raw_pcm=False):
        self.fail = fail or {}
        self.merge_full = merge_full
        self.raw_pcm = raw_pcm
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        model = request["model"]
        if model in self.fail:
            raise self.fail[model]
        texts = [c["text"].replace(v.LINE_SEPARATOR, "") for c in request["input"][0]["content"]]
        gap = 0.1 if self.merge_full and len(texts) > 1 else 0.9
        audio = take(texts, gap=gap)
        blob = (audio.clip(-1, 1) * 32767).astype("<i2").tobytes() if self.raw_pcm else wav_bytes(audio)
        return {"id": "fake", "model": model, "output_audio": {"mime_type": "audio/wav", "data": base64.b64encode(blob).decode()}}

    def generate_content(self, **request):
        """Legacy generate_content TTS path: raw PCM bytes in candidates[0].content.parts[0].inline_data."""
        self.calls.append(request)
        model = request["model"]
        if model in self.fail:
            raise self.fail[model]
        body = request["contents"]
        marker = "\n\nTranscript:\n\n"
        texts = body.split(marker, 1)[1].split("\n\n") if marker in body else body.split("\n\n")[1:]
        gap = 0.1 if self.merge_full and len(texts) > 1 else 0.9
        pcm = (take(texts, gap=gap).clip(-1, 1) * 32767).astype("<i2").tobytes()
        part = SimpleNamespace(inline_data=SimpleNamespace(data=pcm, mime_type="audio/L16;codec=pcm;rate=24000"))
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


class ApiError(Exception):
    def __init__(self, code, status):
        super().__init__(f"{code} {status}")
        self.code = code
        self.status = status


def single_plan(**over):
    plan = {
        "schema_version": "1.0", "duration_seconds": 20,
        "segments": [
            {"start_seconds": 0, "end_seconds": 5, "narration": {"text": "Open the customer lookup app.", "purpose": "introduce", "tone": "friendly", "pace": "normal"}},
            {"start_seconds": 6, "end_seconds": 9, "narration": None, "fit_status": "no_voice"},
            {"start_seconds": 10, "end_seconds": 15, "narration": {"text": "Search for the order number to see its status.", "purpose": "explain_action", "tone": "calm", "pace": "slow"}},
        ],
    }
    plan.update(over)
    return plan


def dialogue_plan():
    return {
        "schema_version": "1.1", "mode": "dialogue", "duration_seconds": 20,
        "speakers": [{"id": "host", "name": "Maya", "role": "host", "voice": "Puck"},
                     {"id": "guest", "name": "Ravi", "role": "expert", "voice": "Kore"}],
        "decision": {"narrate": True, "reason": "demo"},
        "segments": [
            {"start_seconds": 0, "end_seconds": 8, "turns": [
                {"speaker": "host", "text": "What are we looking at here?", "style": "curious", "tone": "friendly", "pace": "normal"},
                {"speaker": "guest", "text": "The order lookup screen.", "tone": "confident", "pace": "brisk"}]},
            {"start_seconds": 12, "end_seconds": 18, "turns": [
                {"speaker": "host", "text": "And that is the status.", "tone": "friendly", "pace": "normal"}]},
        ],
    }


def identity(samples, rate, speed):
    """Processor stand-in: no ffmpeg; naive resample and speed by index stepping."""
    step = rate / v.OUTPUT_RATE * speed
    idx = np.arange(0, samples.size - 1, step)
    return np.interp(idx, np.arange(samples.size), samples).astype(np.float32)


# ------------------------------------------------------------ request building

def test_single_request_shape():
    script = v.read_script(single_plan())
    req = v.build_request(script, "gemini-3.8-flash-tts")
    assert req["model"] == "gemini-3.8-flash-tts"
    assert req["response_format"] == {"type": "audio"}
    assert req["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    content = req["input"][0]["content"]
    assert req["input"][0]["type"] == "user_input"
    assert len(content) == 2
    assert content[0]["text"].endswith("<long pause>") and "<long pause>" not in content[1]["text"]
    meta = content[0]["annotations"][0]
    assert meta == {"type": "speech_metadata", "style": "friendly, normal pace, introducing the topic"}
    assert content[1]["annotations"][0]["style"].startswith("calm, speaking slowly")


def test_dialogue_request_shape():
    script = v.read_script(dialogue_plan())
    req = v.build_request(script, "m")
    assert req["generation_config"]["speech_config"] == {"speakers": [
        {"speaker": "Maya", "voice": "Puck"}, {"speaker": "Ravi", "voice": "Kore"}]}
    metas = [c["annotations"][0] for c in req["input"][0]["content"]]
    assert [m["speaker"] for m in metas] == ["Maya", "Ravi", "Maya"]
    assert metas[0]["style"] == "curious, friendly, normal pace"
    assert [ln.turn_index for ln in script.lines] == [0, 1, 0]


def test_defaults_and_unknown_speaker():
    script = v.read_script({"mode": "dialogue", "duration_seconds": 5, "segments": [
        {"start_seconds": 0, "end_seconds": 4, "turns": [{"speaker": "host", "text": "Hi"}, {"speaker": "guest", "text": "Hello"}]}]})
    assert [s["voice"] for s in script.speakers] == ["Puck", "Kore"]
    with pytest.raises(v.VoiceError):
        v.read_script({"mode": "dialogue", "duration_seconds": 5, "segments": [
            {"start_seconds": 0, "end_seconds": 4, "turns": [{"speaker": "nobody", "text": "Hi"}]}]})


def test_model_list_env():
    assert v.model_list({}) == list(v.DEFAULT_MODELS)
    assert v.model_list({v.MODELS_ENV: "a, b"}) == ["a", "b"]


def test_retryable_detection():
    assert v.is_retryable(ApiError(429, "RESOURCE_EXHAUSTED"))
    assert v.is_retryable(ApiError(503, "UNAVAILABLE"))
    assert v.is_retryable(RuntimeError("500 INTERNAL error"))
    assert not v.is_retryable(ApiError(400, "INVALID_ARGUMENT"))


# ------------------------------------------------------------------ splitter

def test_split_exact_n():
    texts = ["Open the app now please.", "Then search.", "Finally read the status line slowly."]
    pieces = v.split_pieces(take(texts), RATE, texts)
    assert pieces is not None and len(pieces) == 3
    for p, t in zip(pieces, texts):
        assert abs(p.size / RATE - len(t) / CHARS_PER_SECOND) < 0.2


def test_split_picks_longest_gaps_when_extra_pauses():
    texts = ["A first line of words.", "A second line of words."]
    audio = np.concatenate([silence(0.3), speech(0.7), silence(0.55), speech(0.8), silence(1.2), speech(1.5), silence(0.3)])
    pieces = v.split_pieces(audio, RATE, texts)
    assert pieces is not None
    assert pieces[0].size / RATE == pytest.approx(0.7 + 0.55 + 0.8, abs=0.2)


def test_split_mismatch_returns_none():
    texts = ["one line here", "two line here", "three line here"]
    assert v.split_pieces(take(texts, gap=0.1), RATE, texts) is None


# ------------------------------------------------------------- orchestration

def test_single_end_to_end(tmp_path):
    client = FakeTTS()
    res = v.run_voice(single_plan(), tmp_path, client=client, models=["m1", "m2"], processor=identity, log=lambda m: None)
    assert res.written and res.model_used == "m1" and not res.fallback_used
    assert len(client.calls) == 1
    tl = json.loads((tmp_path / "voiceover-timeline.json").read_text())
    assert tl["schema_version"] == "1.0" and tl["model_used"] == "m1" and tl["fallback_used"] is False
    assert [ln["segment_index"] for ln in tl["lines"]] == [0, 2]
    assert tl["lines"][0]["turn_index"] is None and tl["lines"][0]["speaker"] == "narrator"
    assert tl["lines"][0]["start_seconds"] == 0 and tl["lines"][1]["start_seconds"] == 10
    assert all(ln["fit_status"] == "fits" for ln in tl["lines"])
    with wave.open(str(tmp_path / "voiceover.wav")) as wf:
        assert wf.getframerate() == 48000 and wf.getnchannels() == 1
        assert wf.getnframes() == 20 * 48000
    req = json.loads((tmp_path / "tts-request.json").read_text())
    assert '"data"' not in json.dumps(req) and "api_key" not in json.dumps(req)
    resp = json.loads((tmp_path / "tts-response.json").read_text())
    assert resp["response"]["output_audio"]["data"] == "<audio stripped>"


def test_dialogue_placement(tmp_path):
    v.run_voice(dialogue_plan(), tmp_path, client=FakeTTS(), models=["m"], processor=identity, log=lambda m: None)
    lines = json.loads((tmp_path / "voiceover-timeline.json").read_text())["lines"]
    assert [(ln["segment_index"], ln["turn_index"], ln["speaker"]) for ln in lines] == [(0, 0, "host"), (0, 1, "guest"), (1, 0, "host")]
    assert lines[1]["start_seconds"] == pytest.approx(lines[0]["end_seconds"] + v.TURN_GAP, abs=0.002)
    assert lines[2]["start_seconds"] == 12


def test_fallback_revoices_whole_script(tmp_path):
    client = FakeTTS(fail={"primary": ApiError(503, "UNAVAILABLE")})
    logs = []
    res = v.run_voice(dialogue_plan(), tmp_path, client=client, models=["primary", "backup"], processor=identity, log=logs.append)
    assert res.model_used == "backup" and res.fallback_used
    assert [c["model"] for c in client.calls] == ["primary", "backup"]
    assert len(client.calls[1]["input"][0]["content"]) == 3
    assert "Gemini TTS model used: backup" in logs
    tl = json.loads((tmp_path / "voiceover-timeline.json").read_text())
    assert tl["model_used"] == "backup" and tl["fallback_used"] is True
    assert json.loads((tmp_path / "tts-response.json").read_text())["failed_attempts"][0]["model"] == "primary"


def test_non_retryable_error_stops(tmp_path):
    client = FakeTTS(fail={"primary": ApiError(400, "INVALID_ARGUMENT")})
    with pytest.raises(ApiError):
        v.run_voice(single_plan(), tmp_path, client=client, models=["primary", "backup"], processor=identity, log=lambda m: None)
    assert len(client.calls) == 1


def test_split_mismatch_falls_back_per_line(tmp_path):
    client = FakeTTS(merge_full=True)
    logs = []
    res = v.run_voice(dialogue_plan(), tmp_path, client=client, models=["m"], processor=identity, log=logs.append)
    assert res.line_calls == 3 and len(client.calls) == 4
    per_line = client.calls[1:]
    assert all(len(c["input"][0]["content"]) == 1 for c in per_line)
    assert per_line[1]["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    assert any("voicing each line separately" in m for m in logs)
    assert (tmp_path / "tts-line-requests.json").exists()


def test_per_line_cap_stops_and_keeps_take(tmp_path):
    client = FakeTTS(merge_full=True)
    with pytest.raises(v.VoiceError, match="max-line-calls"):
        v.run_voice(dialogue_plan(), tmp_path, client=client, models=["m"], max_line_calls=2, processor=identity, log=lambda m: None)
    assert len(client.calls) == 1
    assert (tmp_path / "tts-full.wav").stat().st_size > 1000
    assert not (tmp_path / "voiceover.wav").exists()


def test_raw_pcm_response(tmp_path):
    res = v.run_voice(single_plan(), tmp_path, client=FakeTTS(raw_pcm=True), models=["m"], processor=identity, log=lambda m: None)
    assert res.written
    assert (tmp_path / "tts-full.wav").read_bytes()[:4] == b"RIFF"


def test_no_narration_writes_nothing(tmp_path):
    client = FakeTTS()
    res = v.run_voice(single_plan(decision={"narrate": False, "reason": "music video"}), tmp_path / "o", client=client, log=lambda m: None)
    assert not res.written and not client.calls and not (tmp_path / "o").exists()
    res = v.run_voice(single_plan(segments=[]), tmp_path / "o", client=client, log=lambda m: None)
    assert not res.written and not (tmp_path / "o").exists()


# ------------------------------------------------------------------ fitting

def _script(segments, duration=10):
    return v.read_script({"duration_seconds": duration, "segments": segments})


def _seg(a, b, text="x" * 30):
    return {"start_seconds": a, "end_seconds": b, "narration": {"text": text, "purpose": "context"}}


def test_fit_speeds_up_within_cap():
    script = _script([_seg(0, 2.0)])
    _, placed = v.fit_and_place(script, [speech(2.1)], RATE, identity)
    assert placed[0].speed == pytest.approx(2.1 / 2.0, abs=1e-3)
    assert placed[0].fit_status == "sped_up"


def test_fit_runs_into_silence_then_overflow():
    script = _script([_seg(0, 2.0), _seg(5, 6)])
    _, placed = v.fit_and_place(script, [speech(3.0), speech(0.5)], RATE, identity)
    assert placed[0].speed == pytest.approx(v.MAX_SPEED)
    assert placed[0].fit_status == "runs_into_silence"
    script = _script([_seg(0, 2.0), _seg(2.5, 4)])
    _, placed = v.fit_and_place(script, [speech(3.0), speech(0.5)], RATE, identity)
    assert placed[0].fit_status == "overflow"
    assert placed[0].end - placed[0].start == pytest.approx(3.0 / v.MAX_SPEED, abs=0.01)


def test_track_exact_length_and_position():
    script = _script([_seg(4, 6)], duration=7.5)
    track, _placed = v.fit_and_place(script, [speech(1.0)], RATE, identity)
    assert track.size == int(7.5 * v.OUTPUT_RATE)
    assert np.abs(track[: int(3.9 * v.OUTPUT_RATE)]).max() == 0
    assert np.abs(track[int(4.1 * v.OUTPUT_RATE): int(4.6 * v.OUTPUT_RATE)]).max() > 0.1


@needs_ffmpeg
def test_process_piece_atempo_and_resample():
    out = v.process_piece(speech(2.2), RATE, 1.10)
    assert out.size / v.OUTPUT_RATE == pytest.approx(2.0, abs=0.03)
    plain = v.process_piece(speech(1.0), RATE, 1.0)
    assert plain.size / v.OUTPUT_RATE == pytest.approx(1.0, abs=0.01)


@needs_ffmpeg
def test_end_to_end_with_ffmpeg(tmp_path):
    res = v.run_voice(single_plan(), tmp_path, client=FakeTTS(), models=["m"], log=lambda m: None)
    assert res.written
    with wave.open(str(tmp_path / "voiceover.wav")) as wf:
        assert wf.getnframes() == 20 * 48000


def test_cli_registers_voice(tmp_path, monkeypatch, capsys):
    import sys

    from narrator import cli
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(single_plan(decision={"narrate": False, "reason": "not useful"})))
    monkeypatch.setattr(sys, "argv", ["narrator", "voice", str(plan), "-o", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert "Narration not requested" in capsys.readouterr().out


# ------------------------------------------------- legacy generate_content path

LEGACY = "gemini-2.5-flash-preview-tts"


def test_legacy_routing():
    script = v.read_script(single_plan())
    assert v.uses_legacy_api(LEGACY) and not v.uses_legacy_api("gemini-3.8-flash-tts")
    assert "contents" in v.build_tts_request(script, LEGACY)
    assert "input" in v.build_tts_request(script, "gemini-3.8-flash-tts")


def test_legacy_single_request_shape():
    script = v.read_script(single_plan())
    req = v.build_legacy_request(script, LEGACY)
    assert req["model"] == LEGACY
    assert req["config"] == {"response_modalities": ["AUDIO"],
                             "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}}}
    body = req["contents"]
    assert "Line 1: friendly, normal pace, introducing the topic." in body
    assert body.endswith("Open the customer lookup app.\n\nSearch for the order number to see its status.")
    assert "<long pause>" not in body


def test_legacy_dialogue_request_shape():
    script = v.read_script(dialogue_plan())
    req = v.build_legacy_request(script, LEGACY)
    cfgs = req["config"]["speech_config"]["multi_speaker_voice_config"]["speaker_voice_configs"]
    assert cfgs == [{"speaker": "Maya", "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
                    {"speaker": "Ravi", "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}}]
    assert "conversation between Maya and Ravi" in req["contents"]
    assert "Maya: What are we looking at here?\n\nRavi: The order lookup screen." in req["contents"]
    one = v.build_legacy_request(script, LEGACY, [script.lines[1]])
    assert one["config"]["speech_config"] == {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}}
    assert one["contents"].endswith("The order lookup screen.")


def test_legacy_config_validates_against_sdk_types():
    types = pytest.importorskip("google.genai.types")
    for plan in (single_plan(), dialogue_plan()):
        req = v.build_legacy_request(v.read_script(plan), LEGACY)
        types.GenerateContentConfig.model_validate(req["config"])


def test_legacy_audio_extraction():
    pcm = (np.ones(160) * 0.5 * 32767).astype("<i2").tobytes()
    resp = {"candidates": [{"content": {"parts": [{"text": "x"}, {"inline_data": {"data": base64.b64encode(pcm).decode(), "mime_type": "audio/L16;codec=pcm;rate=16000"}}]}}]}
    blob, samples, rate = v.decode_response(resp)
    assert blob == pcm and rate == 16000 and samples.size == 160


def test_fallback_to_legacy_model_revoices_whole_script(tmp_path):
    client = FakeTTS(fail={"gemini-3.8-flash-tts": ApiError(429, "RESOURCE_EXHAUSTED")})
    logs = []
    res = v.run_voice(dialogue_plan(), tmp_path, client=client, processor=identity, log=logs.append,
                      models=["gemini-3.8-flash-tts", LEGACY])
    assert res.model_used == LEGACY and res.fallback_used
    assert "input" in client.calls[0] and "contents" in client.calls[1] and len(client.calls) == 2
    tl = json.loads((tmp_path / "voiceover-timeline.json").read_text())
    assert tl["model_used"] == LEGACY and len(tl["lines"]) == 3
    assert json.loads((tmp_path / "tts-request.json").read_text())["model"] == LEGACY
    assert "Gemini TTS model used: " + LEGACY in logs


def test_legacy_per_line_fallback(tmp_path):
    client = FakeTTS(merge_full=True)
    res = v.run_voice(dialogue_plan(), tmp_path, client=client, models=[LEGACY], processor=identity, log=lambda m: None)
    assert res.line_calls == 3 and all("contents" in c for c in client.calls)
