---
name: narrator
description: Decide whether a video should get a voiceover and create timestamped, evidence-grounded narration plans (single narrator or two-voice dialogue) from video-analysis output.
---

# Narrator

Create narration scripts for videos, especially demos, tutorials, walkthroughs, presentations and screen recordings. Narrator consumes the `analysis.json` produced by video-analysis and produces a timed script (`narration-plan.json`). Audio is a separate stage (`narrator voice`).

## Pipeline

```
video -> video-analysis -> analysis.json
      -> goldilock-av-mixer detect-speech -> speech.json        (optional)
analysis.json -> narrator decide                                 (no API call)
analysis.json -> narrator plan -> narration-plan.json            (one Gemini call, or none)
narration-plan.json -> narrator voice -> voiceover.wav + voiceover-timeline.json
voiceover + original + music bed -> goldilock-av-mixer mix -> final.mp4
```

## Workflow

1. Read the video type from `video.dimensions.video_type`, falling back to `video.dimensions.semantic.video_type`. Missing means `unknown` with confidence 0. The analyser's classification is authoritative; this skill never reclassifies.
2. Decide (in code, no model call): narrate when the type is one of demo, tutorial, walkthrough, presentation, documentation, screen_recording, review and confidence is at least 0.6. Otherwise do not narrate, unless the person passes `--force` or an explicit `--mode`. When the decision is "no", the plan is written with `decision.narrate=false`, empty segments, and Gemini is not called.
3. Build the planning context: duration, video type, semantic timeline, scenes, screen data, and `semantic.screen_actions` when present.
4. Add the person's notes (`--notes`) as extra context: product names, audience, emphasis. Notes never override the evidence rule; a claim in the notes that the video does not show is not narrated.
5. Mark original speech (`--speech-segments speech.json`) as `no_voice_windows` with reason `original_speech`. The prompt says not to narrate over them; any segment that still overlaps is flagged.
6. Generate concise spoken narration only for useful visual windows. Tie every segment to timestamped evidence. Do not narrate every click.
7. After the model returns, code recomputes speech estimates and fit, and flags problems in `review_notes`. Nothing is clamped or dropped silently.
8. Never infer functionality, outcomes, UI labels, people, actions or intent not demonstrated by observable evidence.

## Modes

- `single` (default): one narrator, voice `Kore`. Segments carry `narration`.
- `dialogue`: two speakers, by default a curious host (`Puck`) and a guide (`Kore`). Segments carry `turns: [{speaker, text, style, tone, pace}]`. Both speakers stay grounded in on-screen evidence, with no filler banter and no repeating each other. Dialogue needs more words, so it uses fewer, slightly longer moments. Names and roles: `--speakers "Maya:curious host,Arun:guide"`; voices: `--voices Puck,Kore` (Gemini TTS prebuilt voice names). The mode is called `dialogue` so it is not confused with the video type `conversation`.

## Deterministic checks

- `estimated_speech_seconds` = words / words-per-minute x 60, where words-per-minute is scaled by pace (slow 0.85, normal 1.0, brisk 1.15). Dialogue: sum of turns plus a 0.3 s gap between consecutive turns.
- `fit_status`: `fits` when the estimate is at most 90% of the window, `tight` when it is above 90% and at most 100%, `too_long` above 100%. Segments without speech keep `no_voice` or `needs_review`.
- Overlapping segments and segments that overlap original speech get a `review_reason` and a line in `review_notes`.

## Commands

```bash
narrator decide analysis.json [--mode single|dialogue] [--force]
narrator plan analysis.json -o narration-plan.json \
  [--mode single|dialogue] [--force] [--notes "..."] \
  [--speech-segments speech.json] [--speakers "Name:role,..."] [--voices Puck,Kore] \
  [--model gemini-3.8-flash] [--language en] [--audience general] [--tone friendly] [--words-per-minute 145]
narrator inspect analysis.json
```

Model fallback: the requested model, then `NARRATOR_MODELS` (comma-separated) or the default chain gemini-3.8-flash, gemini-4-hydrogen, gemini-3.6-flash, gemini-2.5-flash, gemini-3.5-flash-lite. Only quota, rate-limit and server errors move to the next model. At most `NARRATOR_MAX_MODEL_ATTEMPTS` models are tried (default 3); `NARRATOR_MODEL_FALLBACK=0` turns fallback off. The model used is printed as `Gemini model used: X`.

## Input

- Required: completed `analysis.json` from video-analysis.
- Optional: `speech.json` from `goldilock-av-mixer detect-speech` (a list of `{start_seconds, end_seconds}`).
- Optional preferences: mode, speakers, voices, notes, language, audience, tone, speaking rate.

## Output

Next to the plan:

- `narration-plan.json`: NarrationPlan 1.1 (`mode`, `speakers`, `decision`, segments, no-voice windows, review notes). 1.0 files still validate.
- `narration-request.json`: the prompt sent to Gemini (no secrets). Only when Gemini is called.
- `narration-response.raw.json`: the raw response text. Only when Gemini is called; written even if the response fails validation.

See `references/narration-plan.schema.json`, `examples/narration-plan.json` and `examples/narration-plan.dialogue.json`.

## Voice stage

`narrator voice narration-plan.json` (owned by the narrator-voice work) turns the plan into `voiceover.wav` (48 kHz mono, exactly the video duration) and `voiceover-timeline.json`. Pass both to goldilock-av-mixer `mix --voiceover ... --voice-timeline ...`, and the timeline to the music-bed composer.
