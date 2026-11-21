"""Voiceover stage: NarrationPlan -> voiceover.wav + voiceover-timeline.json (contract C).

The whole script is voiced in ONE Gemini TTS request (Interactions API). The returned
audio is split at the `<long pause>` gaps, each piece is fitted to its slot (ffmpeg
atempo up to 1.10) and placed on a silent 48 kHz mono track as long as the video.

The plan is read as plain JSON so NarrationPlan 1.0 and 1.1 both work.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

DEFAULT_MODELS = ("gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts")
MODELS_ENV = "NARRATOR_TTS_MODELS"
OUTPUT_RATE = 48000
TTS_RATE = 24000
MAX_SPEED = 1.10
TURN_GAP = 0.25
LINE_SEPARATOR = " <long pause>"
DEFAULT_MAX_LINE_CALLS = 12
FRAME_SECONDS = 0.02
MIN_GAP_SECONDS = 0.5
EDGE_PAD_SECONDS = 0.05

DEFAULT_SINGLE_SPEAKERS = [{"id": "narrator", "name": "Narrator", "role": "narrator", "voice": "Kore"}]
DEFAULT_DIALOGUE_SPEAKERS = [
    {"id": "host", "name": "Host", "role": "host", "voice": "Puck"},
    {"id": "guest", "name": "Guest", "role": "guest", "voice": "Kore"},
]

PURPOSE_PHRASES = {
    "introduce": "introducing the topic",
    "explain_action": "explaining what is happening",
    "explain_result": "explaining the result",
    "transition": "moving on",
    "summarize": "wrapping up",
    "warn": "pointing out a caution",
    "context": "giving context",
}
PACE_PHRASES = {"slow": "speaking slowly", "normal": "normal pace", "brisk": "speaking briskly"}
RETRYABLE_MARKERS = ("INTERNAL", "UNAVAILABLE", "RESOURCE_EXHAUSTED")


class VoiceError(RuntimeError):
    """Voiceover could not be produced."""


class RetryableTTSError(VoiceError):
    """A TTS failure that should move on to the next model."""


# ---------------------------------------------------------------- plan reading

@dataclass
class Line:
    segment_index: int
    turn_index: int | None
    speaker: str  # speaker id
    text: str
    style: str
    segment_start: float
    segment_end: float


@dataclass
class Script:
    mode: str
    speakers: list[dict[str, Any]]
    duration_seconds: float
    lines: list[Line]
    narrate: bool = True
    reason: str = ""

    def label(self, speaker_id: str) -> str:
        """Speaker label sent to TTS (name if present and unique, else id)."""
        names = [s.get("name") or s["id"] for s in self.speakers]
        for spk in self.speakers:
            if spk["id"] == speaker_id:
                name = spk.get("name") or spk["id"]
                return name if names.count(name) == 1 else spk["id"]
        return speaker_id

    def voice(self, speaker_id: str) -> str:
        for spk in self.speakers:
            if spk["id"] == speaker_id:
                return spk["voice"]
        return self.speakers[0]["voice"]


def build_style(tone: str | None, pace: str | None, purpose: str | None = None, extra: str | None = None) -> str:
    parts = []
    if extra:
        parts.append(str(extra).strip())
    parts.append(tone or "neutral")
    parts.append(PACE_PHRASES.get(pace or "normal", f"{pace} pace"))
    if purpose and purpose in PURPOSE_PHRASES:
        parts.append(PURPOSE_PHRASES[purpose])
    return ", ".join(p for p in parts if p)


def _speakers(plan: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    defaults = DEFAULT_DIALOGUE_SPEAKERS if mode == "dialogue" else DEFAULT_SINGLE_SPEAKERS
    raw = plan.get("speakers") or []
    out = []
    for i, spk in enumerate(raw):
        if not isinstance(spk, dict):
            continue
        base = defaults[min(i, len(defaults) - 1)]
        sid = str(spk.get("id") or spk.get("name") or base["id"])
        out.append({"id": sid, "name": spk.get("name") or sid, "role": spk.get("role") or base["role"],
                    "voice": spk.get("voice") or base["voice"]})
    if not out:
        out = [dict(s) for s in defaults]
    limit = 2 if mode == "dialogue" else 1
    if len(out) > limit:
        raise VoiceError(f"{mode} mode allows {limit} speaker(s), plan has {len(out)}")
    return out


def read_script(plan: dict[str, Any]) -> Script:
    """Turn a NarrationPlan 1.0/1.1 dict into an ordered list of voiced lines."""
    mode = plan.get("mode") or "single"
    if mode not in {"single", "dialogue"}:
        raise VoiceError(f"unknown narration mode: {mode}")
    speakers = _speakers(plan, mode)
    decision = plan.get("decision") or {}
    narrate = decision.get("narrate", True) is not False
    duration = float(plan.get("duration_seconds") or 0)
    if duration <= 0:
        raise VoiceError("plan duration_seconds must be > 0")
    ids = {s["id"] for s in speakers}
    lines: list[Line] = []
    for si, seg in enumerate(plan.get("segments") or []):
        start, end = float(seg["start_seconds"]), float(seg["end_seconds"])
        if mode == "dialogue" and seg.get("turns"):
            for ti, turn in enumerate(seg["turns"]):
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                spk = str(turn.get("speaker") or speakers[0]["id"])
                if spk not in ids:
                    raise VoiceError(f"segment {si} turn {ti}: unknown speaker {spk!r}")
                lines.append(Line(si, ti, spk, text, build_style(turn.get("tone"), turn.get("pace"), None, turn.get("style")), start, end))
            continue
        narration = seg.get("narration")
        if not narration or not (narration.get("text") or "").strip():
            continue
        lines.append(Line(si, None, speakers[0]["id"], narration["text"].strip(),
                          build_style(narration.get("tone"), narration.get("pace"), narration.get("purpose")), start, end))
    return Script(mode, speakers, duration, lines, narrate, str(decision.get("reason") or ""))


# ------------------------------------------------------------- request building

def build_request(script: Script, model: str, lines: list[Line] | None = None) -> dict[str, Any]:
    """One Interactions API request for the given lines (default: the whole script).

    Lines are separate text items, each with speech_metadata, separated by <long pause>.
    Two distinct speakers use the multi-speaker config object; otherwise the
    single-speaker list form is used (shapes per the Gemini speech-generation doc).
    """
    lines = script.lines if lines is None else lines
    used = []
    for ln in lines:
        if ln.speaker not in used:
            used.append(ln.speaker)
    multi = script.mode == "dialogue" and len(used) > 1
    content = []
    for i, ln in enumerate(lines):
        meta: dict[str, Any] = {"type": "speech_metadata", "style": ln.style}
        if multi:
            meta["speaker"] = script.label(ln.speaker)
        text = ln.text + (LINE_SEPARATOR if i < len(lines) - 1 else "")
        content.append({"type": "text", "text": text, "annotations": [meta]})
    if multi:
        speech_config: Any = {"speakers": [{"speaker": script.label(s["id"]), "voice": s["voice"]}
                                           for s in script.speakers if s["id"] in used]}
    else:
        speech_config = [{"voice": script.voice(used[0] if used else script.speakers[0]["id"])}]
    return {
        "model": model,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": speech_config},
    }


# Models served through the older generate_content TTS path rather than the Interactions
# API. gemini-2.5-flash-preview-tts is listed on the models page but not on the
# speech-generation (Interactions) page. Shape checked against google-genai 3.6.1 types
# (GenerateContentConfig.speech_config), not against the live API.
LEGACY_MODELS = frozenset({"gemini-2.5-flash-preview-tts"})


def uses_legacy_api(model: str) -> bool:
    return model in LEGACY_MODELS


def build_legacy_request(script: Script, model: str, lines: list[Line] | None = None) -> dict[str, Any]:
    """generate_content TTS request: a plain-text transcript plus speech_config.

    There are no per-line annotations on this path, so styles go into a short
    direction block before the transcript, and lines are separated by blank lines
    with an instruction to pause between them. Dialogue lines are prefixed with
    the speaker label that matches multi_speaker_voice_config.
    """
    lines = script.lines if lines is None else lines
    used = []
    for ln in lines:
        if ln.speaker not in used:
            used.append(ln.speaker)
    multi = script.mode == "dialogue" and len(used) > 1
    directions = []
    transcript = []
    for i, ln in enumerate(lines, 1):
        who = f"{script.label(ln.speaker)}: " if multi else ""
        directions.append(f"Line {i}: {ln.style}.")
        transcript.append(f"{who}{ln.text}")
    if len(lines) == 1:
        head = f"Say this in a {lines[0].style} delivery:"
    else:
        names = " and ".join(script.label(s) for s in used)
        kind = f"the following conversation between {names}" if multi else "the following lines"
        head = (f"Read {kind} aloud. Leave a long pause of about one second between lines. "
                "Speak only the transcript, not these directions.\n" + "\n".join(directions) + "\n\nTranscript:")
    contents = head + "\n\n" + "\n\n".join(transcript)
    if multi:
        speech_config: dict[str, Any] = {"multi_speaker_voice_config": {"speaker_voice_configs": [
            {"speaker": script.label(s["id"]), "voice_config": {"prebuilt_voice_config": {"voice_name": s["voice"]}}}
            for s in script.speakers if s["id"] in used]}}
    else:
        voice = script.voice(used[0] if used else script.speakers[0]["id"])
        speech_config = {"voice_config": {"prebuilt_voice_config": {"voice_name": voice}}}
    return {"model": model, "contents": contents,
            "config": {"response_modalities": ["AUDIO"], "speech_config": speech_config}}


def build_tts_request(script: Script, model: str, lines: list[Line] | None = None) -> dict[str, Any]:
    """Pick the request shape for the model's API path."""
    return (build_legacy_request if uses_legacy_api(model) else build_request)(script, model, lines)


# ------------------------------------------------------------------ TTS client

class TTSClient(Protocol):
    def create(self, **request: Any) -> Any: ...  # Interactions API
    def generate_content(self, **request: Any) -> Any: ...  # legacy TTS path


class GeminiTTSClient:
    """Thin wrapper over google-genai `client.interactions.create`."""

    def __init__(self, api_key: str | None = None):
        try:
            from google import genai
        except ImportError as exc:
            raise VoiceError("Install narrator[gemini] for Gemini TTS") from exc
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise VoiceError("GEMINI_API_KEY is required")
        self._client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})

    def create(self, **request: Any) -> Any:
        return self._client.interactions.create(**request)

    def generate_content(self, **request: Any) -> Any:
        return self._client.models.generate_content(**request)


def call_tts(client: TTSClient, request: dict[str, Any]) -> Any:
    if "contents" in request:
        return client.generate_content(**request)
    return client.create(**request)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RetryableTTSError):
        return True
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and (val == 429 or 500 <= val < 600):
            return True
    text = f"{getattr(exc, 'status', '') or ''} {exc}".upper()
    return any(m in text for m in RETRYABLE_MARKERS)


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _legacy_inline_audio(response: Any) -> tuple[Any, str]:
    for cand in _get(response, "candidates") or []:
        content = _get(cand, "content")
        for part in (_get(content, "parts") or []) if content is not None else []:
            inline = _get(part, "inline_data")
            if inline is not None and _get(inline, "data"):
                return _get(inline, "data"), str(_get(inline, "mime_type") or "")
    return None, ""


def _rate_from_mime(mime: str) -> int | None:
    for piece in mime.replace(";", " ").split():
        if piece.lower().startswith("rate="):
            try:
                return int(piece.split("=", 1)[1])
            except ValueError:
                return None
    return None


def extract_audio_and_rate(response: Any) -> tuple[bytes, int | None]:
    """Audio bytes from an Interactions response (output_audio) or a generate_content
    response (candidates[].content.parts[].inline_data), plus a sample rate if declared."""
    audio = _get(response, "output_audio")
    data = _get(audio, "data") if audio is not None else None
    rate = _rate_from_mime(str(_get(audio, "mime_type") or "")) if audio is not None else None
    if not data:
        data, mime = _legacy_inline_audio(response)
        rate = _rate_from_mime(mime)
    if not data:
        raise RetryableTTSError("TTS response has no audio data")
    blob = base64.b64decode(data) if isinstance(data, str) else bytes(data)
    return blob, rate


def extract_audio(response: Any) -> bytes:
    return extract_audio_and_rate(response)[0]


def decode_response(response: Any) -> tuple[bytes, np.ndarray, int]:
    blob, rate = extract_audio_and_rate(response)
    samples, got = decode_audio(blob, rate or TTS_RATE)
    return blob, samples, got


def _strip_audio(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<audio stripped>" if k == "data" and isinstance(v, (str, bytes)) else _strip_audio(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_audio(v) for v in value]
    return value


def response_metadata(response: Any) -> Any:
    for meth in ("model_dump", "to_dict", "dict"):
        fn = getattr(response, meth, None)
        if callable(fn):
            try:
                dumped = fn(mode="json") if meth == "model_dump" else fn()
                return _strip_audio(json.loads(json.dumps(dumped, default=str)))
            except Exception:  # noqa: BLE001, S112 - metadata is best effort
                continue
    if isinstance(response, dict):
        return _strip_audio(json.loads(json.dumps(response, default=str)))
    return {"repr": type(response).__name__}


def model_list(env: dict[str, str] | None = None) -> list[str]:
    raw = (env if env is not None else os.environ).get(MODELS_ENV, "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or list(DEFAULT_MODELS)


# -------------------------------------------------------------- audio helpers

def decode_audio(blob: bytes, default_rate: int = TTS_RATE) -> tuple[np.ndarray, int]:
    """WAV (RIFF) or headerless 16-bit mono PCM -> float32 mono in [-1, 1]."""
    if blob[:4] == b"RIFF":
        with wave.open(io.BytesIO(blob)) as wf:
            rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if width != 2:
            raise VoiceError(f"unsupported WAV sample width {width}")
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)
        return pcm, rate
    usable = len(blob) - (len(blob) % 2)
    return np.frombuffer(blob[:usable], dtype="<i2").astype(np.float32) / 32768.0, default_rate


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def frame_rms(samples: np.ndarray, rate: int, frame_seconds: float = FRAME_SECONDS) -> np.ndarray:
    n = max(1, round(rate * frame_seconds))
    count = len(samples) // n
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: count * n].reshape(count, n)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def silence_mask(samples: np.ndarray, rate: int) -> np.ndarray:
    rms = frame_rms(samples, rate)
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    floor = float(np.percentile(rms, 5))
    peak = float(np.percentile(rms, 99))
    threshold = floor + 0.08 * (peak - floor)
    return rms <= threshold if peak > floor else np.ones_like(rms, dtype=bool)


def find_gaps(samples: np.ndarray, rate: int, min_gap: float = MIN_GAP_SECONDS) -> list[tuple[float, float]]:
    """Interior silent runs of at least `min_gap` seconds, as (start, end) in seconds."""
    mask = silence_mask(samples, rate)
    gaps, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if i > 0 and j < n and (j - i) * FRAME_SECONDS >= min_gap:
            gaps.append((i * FRAME_SECONDS, j * FRAME_SECONDS))
        i = j
    return gaps


def trim_silence(samples: np.ndarray, rate: int, pad: float = EDGE_PAD_SECONDS) -> np.ndarray:
    mask = silence_mask(samples, rate)
    voiced = np.flatnonzero(~mask)
    if voiced.size == 0:
        return samples[:0]
    frame = round(rate * FRAME_SECONDS)
    p = round(rate * pad)
    start = max(0, voiced[0] * frame - p)
    end = min(len(samples), (voiced[-1] + 1) * frame + p)
    return samples[start:end]


def split_pieces(samples: np.ndarray, rate: int, texts: list[str]) -> list[np.ndarray] | None:
    """Cut the full take into exactly len(texts) pieces, or None if not plausible."""
    n = len(texts)
    if n == 1:
        piece = trim_silence(samples, rate)
        return [piece] if piece.size else None
    gaps = find_gaps(samples, rate)
    if len(gaps) < n - 1:
        return None
    chosen = sorted(sorted(gaps, key=lambda g: (-(g[1] - g[0]), g[0]))[: n - 1])
    cuts = [0] + [round((a + b) / 2 * rate) for a, b in chosen] + [len(samples)]
    pieces = [trim_silence(samples[cuts[k]:cuts[k + 1]], rate) for k in range(n)]
    if any(p.size < int(0.15 * rate) for p in pieces):
        return None
    # Loose plausibility: each piece's share of speech time vs its share of characters.
    total_t = sum(p.size for p in pieces)
    total_c = sum(max(1, len(t)) for t in texts)
    for p, t in zip(pieces, texts):
        ratio = (p.size / total_t) / (max(1, len(t)) / total_c)
        if not 0.2 <= ratio <= 5.0:
            return None
    return pieces


def process_piece(samples: np.ndarray, rate: int, speed: float, out_rate: int = OUTPUT_RATE) -> np.ndarray:
    """Apply ffmpeg atempo (if speed > 1) and resample to out_rate mono."""
    if shutil.which("ffmpeg") is None:
        raise VoiceError("ffmpeg is required for the voiceover stage")
    filters = [f"atempo={speed:.4f}"] if speed > 1.0005 else []
    with tempfile.TemporaryDirectory(prefix="narrator-voice-") as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        write_wav(src, samples, rate)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
        if filters:
            cmd += ["-filter:a", ",".join(filters)]
        cmd += ["-ar", str(out_rate), "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise VoiceError(f"ffmpeg failed: {proc.stderr.strip()[:300]}")
        out, got_rate = decode_audio(dst.read_bytes())
    if got_rate != out_rate:
        raise VoiceError(f"ffmpeg returned {got_rate} Hz, expected {out_rate}")
    return out


# ------------------------------------------------------------ fit and place

@dataclass
class Placed:
    line: Line
    start: float
    end: float
    speed: float
    fit_status: str
    audio: np.ndarray = field(repr=False)


def fit_and_place(script: Script, pieces: list[np.ndarray], rate: int,
                  processor: Callable[[np.ndarray, int, float], np.ndarray] | None = None) -> tuple[np.ndarray, list[Placed]]:
    """Speed-fit pieces per segment, place them, and return the track plus placements.

    fit_status: fits | sped_up | runs_into_silence | overflow.
    """
    processor = processor or (lambda s, r, f: process_piece(s, r, f))
    groups: dict[int, list[int]] = {}
    for idx, ln in enumerate(script.lines):
        groups.setdefault(ln.segment_index, []).append(idx)
    order = sorted(groups)
    seg_starts = [script.lines[groups[s][0]].segment_start for s in order]

    placed: list[Placed] = []
    for gi, seg in enumerate(order):
        idxs = groups[seg]
        first = script.lines[idxs[0]]
        avail = first.segment_end - first.segment_start - TURN_GAP * (len(idxs) - 1)
        speech = sum(pieces[i].size / rate for i in idxs)
        speed = MAX_SPEED if avail <= 0 else min(MAX_SPEED, max(1.0, speech / avail))
        limit = seg_starts[gi + 1] if gi + 1 < len(order) else script.duration_seconds
        cursor = first.segment_start
        for k, i in enumerate(idxs):
            audio = processor(pieces[i], rate, speed)
            start = cursor if k == 0 else cursor + TURN_GAP
            end = start + audio.size / OUTPUT_RATE
            if end <= first.segment_end + 1e-6:
                status = "sped_up" if speed > 1.0005 else "fits"
            elif end <= limit + 1e-6:
                status = "runs_into_silence"
            else:
                status = "overflow"
            placed.append(Placed(script.lines[i], start, end, speed, status, audio))
            cursor = end

    total = round(script.duration_seconds * OUTPUT_RATE)
    track = np.zeros(total, dtype=np.float32)
    for p in placed:
        a = round(p.start * OUTPUT_RATE)
        if a >= total:
            continue
        chunk = p.audio[: total - a]
        track[a:a + chunk.size] += chunk
    return np.clip(track, -1.0, 1.0), placed


def timeline_json(script: Script, placed: list[Placed], model: str, fallback_used: bool) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "duration_seconds": script.duration_seconds,
        "model_used": model,
        "fallback_used": fallback_used,
        "lines": [{
            "segment_index": p.line.segment_index,
            "turn_index": p.line.turn_index,
            "speaker": p.line.speaker,
            "text": p.line.text,
            "start_seconds": round(p.start, 3),
            "end_seconds": round(min(p.end, script.duration_seconds), 3),
            "speed_factor": round(p.speed, 4),
            "fit_status": p.fit_status,
        } for p in placed],
    }


# ---------------------------------------------------------------- orchestration

@dataclass
class VoiceResult:
    written: bool
    message: str
    model_used: str | None = None
    fallback_used: bool = False
    line_calls: int = 0
    overflow: list[int] = field(default_factory=list)


def _voice_on_model(client: TTSClient, script: Script, model: str, out: Path, max_line_calls: int,
                    log: Callable[[str], None]) -> tuple[list[np.ndarray], int, int]:
    request = build_tts_request(script, model)
    (out / "tts-request.json").write_text(json.dumps(request, indent=2))
    response = call_tts(client, request)
    (out / "tts-response.json").write_text(json.dumps({"model": model, "response": response_metadata(response)}, indent=2))
    blob, samples, rate = decode_response(response)
    if blob[:4] == b"RIFF":
        (out / "tts-full.wav").write_bytes(blob)
    else:
        write_wav(out / "tts-full.wav", samples, rate)
    pieces = split_pieces(samples, rate, [ln.text for ln in script.lines])
    if pieces is not None:
        return pieces, rate, 0
    n = len(script.lines)
    found = len(find_gaps(samples, rate))
    if n > max_line_calls:
        raise VoiceError(f"could not split the take into {n} lines (found {found} pauses) and per-line "
                         f"fallback needs {n} calls, over --max-line-calls {max_line_calls}. "
                         f"Full take kept at {out / 'tts-full.wav'}")
    log(f"Could not split the take into {n} lines (found {found} pauses); voicing each line separately ({n} calls).")
    pieces, line_requests, line_rate = [], [], rate
    for ln in script.lines:
        req = build_tts_request(script, model, [ln])
        line_requests.append(req)
        _, s, r = decode_response(call_tts(client, req))
        if r != line_rate:
            raise VoiceError("per-line TTS responses returned different sample rates")
        piece = trim_silence(s, r)
        if piece.size == 0:
            raise RetryableTTSError(f"empty audio for segment {ln.segment_index}")
        pieces.append(piece)
    (out / "tts-line-requests.json").write_text(json.dumps(line_requests, indent=2))
    return pieces, line_rate, n


def run_voice(plan: dict[str, Any] | str | Path, out_dir: str | Path = ".", *, client: TTSClient | None = None,
              models: list[str] | None = None, max_line_calls: int = DEFAULT_MAX_LINE_CALLS,
              log: Callable[[str], None] = print,
              processor: Callable[[np.ndarray, int, float], np.ndarray] | None = None) -> VoiceResult:
    if not isinstance(plan, dict):
        plan = json.loads(Path(plan).read_text())
    script = read_script(plan)
    if not script.narrate:
        msg = f"Narration not requested ({script.reason or 'decision.narrate is false'}); no voiceover written."
        log(msg)
        return VoiceResult(False, msg)
    if not script.lines:
        msg = "Plan has no narrated lines; no voiceover written."
        log(msg)
        return VoiceResult(False, msg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = client or GeminiTTSClient()
    models = models or model_list()
    attempts: list[dict[str, str]] = []
    for mi, model in enumerate(models):
        try:
            pieces, rate, line_calls = _voice_on_model(client, script, model, out, max_line_calls, log)
        except Exception as exc:
            if not is_retryable(exc):
                raise
            attempts.append({"model": model, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            log(f"Gemini TTS model {model} failed ({type(exc).__name__}); "
                + ("re-voicing the whole script on the next model." if mi + 1 < len(models) else "no models left."))
            continue
        fallback_used = mi > 0
        log(f"Gemini TTS model used: {model}")
        if attempts:
            meta_path = out / "tts-response.json"
            meta = json.loads(meta_path.read_text())
            meta["failed_attempts"] = attempts
            meta_path.write_text(json.dumps(meta, indent=2))
        track, placed = fit_and_place(script, pieces, rate, processor)
        write_wav(out / "voiceover.wav", track, OUTPUT_RATE)
        timeline = timeline_json(script, placed, model, fallback_used)
        (out / "voiceover-timeline.json").write_text(json.dumps(timeline, indent=2))
        overflow = [i for i, p in enumerate(placed) if p.fit_status == "overflow"]
        for i in overflow:
            p = placed[i]
            log(f"Overflow: segment {p.line.segment_index} line runs to {p.end:.2f}s and overlaps the next line or the video end.")
        return VoiceResult(True, f"Voiceover written to {out / 'voiceover.wav'}", model, fallback_used, line_calls, overflow)
    raise VoiceError("all Gemini TTS models failed: " + "; ".join(f"{a['model']}: {a['error']}" for a in attempts))
