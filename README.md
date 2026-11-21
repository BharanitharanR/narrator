# narrator

An Agent Skill for turning video understanding into timestamped AI voiceover scripts.

Narrator consumes [video-analysis](https://github.com/BharanitharanR/video-analysis) output. The analyser identifies video type; narrator uses that classification and semantic timeline to create evidence-grounded narration.

Pipeline: video -> video-analysis -> analysis.json -> narrator -> narration-plan.json -> TTS -> voiceover.wav -> goldilock-av-mixer -> final.mp4.

V1 generates concise narration segments with timestamps, evidence references, confidence, estimated speech duration, fit checks and no-voice windows. TTS is a separate `voice` stage (below); audio mixing is done by goldilock-av-mixer.

Narration is explanatory, not speculative. Each claim should be traceable to timestamped observations.

## Voiceover (`narrator voice`)

```bash
narrator voice narration-plan.json -o out/ [--max-line-calls 12]
```

Reads a NarrationPlan (1.0, or 1.1 with `mode`, `speakers`, `decision`) as plain JSON and writes:

- `voiceover.wav`: 48 kHz mono, 16-bit, exactly `duration_seconds` long, silence outside narrated lines.
- `voiceover-timeline.json`: `{schema_version: "1.0", duration_seconds, model_used, fallback_used, lines: [{segment_index, turn_index, speaker, text, start_seconds, end_seconds, speed_factor, fit_status}]}`. `turn_index` is null in single mode.
- `tts-request.json` (request without key), `tts-response.json` (response metadata, audio stripped, plus `failed_attempts` if a model failed), `tts-full.wav` (the raw take, kept for inspection). `tts-line-requests.json` only if the per-line fallback ran.

If `decision.narrate` is false or there are no narrated lines, it prints a message, writes nothing and exits 0.

How it works:

1. The whole script goes to Gemini TTS in one request (Interactions API, `client.interactions.create`). Each line or dialogue turn is one text item with `speech_metadata` (speaker in dialogue mode, style from tone, pace and purpose), and lines are separated by `<long pause>`. Single mode uses one prebuilt voice (default Kore); dialogue uses two (defaults Puck and Kore).
2. Models are tried in order: `gemini-3.8-flash-tts`, then `gemini-2.5-flash-preview-tts`, on retryable errors (429, 5xx, INTERNAL, UNAVAILABLE, RESOURCE_EXHAUSTED). A fallback re-voices the whole script, so one voiceover never mixes models. Override with `NARRATOR_TTS_MODELS=model-a,model-b`. The model used is printed as `Gemini TTS model used: X`.
3. The take is split at its N-1 longest pauses (20 ms RMS frames, silence threshold relative to the noise floor, pauses of at least 0.5 s). If exactly N plausible pieces cannot be found, each line is voiced separately on the same model, up to `--max-line-calls`. Over the cap it stops with an error and keeps `tts-full.wav`.
4. Each segment's lines are placed from the segment start (dialogue turns 0.25 s apart). Lines too long for the segment are sped up with ffmpeg `atempo` up to 1.10. `fit_status` is `fits`, `sped_up`, `runs_into_silence` (ends after the segment but before the next line), or `overflow` (overlaps the next line or the video end; reported, words are not cut, but audio past the video end is lost).

Needs `ffmpeg` on PATH, `GEMINI_API_KEY`, and `pip install narrator[gemini]`. Tests use a fake TTS client and make no network calls.
