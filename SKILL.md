---
name: narrator
description: Detect demonstration-style videos and create timestamped, evidence-grounded voiceover plans using video-analysis output.
---

# Narrator

Create narration scripts for videos, especially demos, tutorials, walkthroughs, presentations and screen recordings. Narrator consumes the `analysis.json` produced by video-analysis and produces a timed script. It does not generate audio.

## Workflow

1. Read `video.dimensions.video_type`, the semantic timeline and the deterministic scene information.
2. Use the analyser's video-type classification as authoritative. Do not reclassify the video in this skill.
3. Generate concise spoken narration only for useful visual windows. Tie every segment to timestamped evidence.
4. Preserve no-voice windows for original speech, important visual reading, transitions and ambient sound.
5. Estimate spoken duration at the configured words-per-minute rate. Text must fit the available visual window; flag tight/overflow segments for review.
6. Never infer functionality, outcomes, UI labels, people, actions or intent not demonstrated by observable evidence.

## Input

- Required: completed `analysis.json` from video-analysis.
- Optional: original video, when the analysis lacks screen/action detail or deeper inspection is requested.
- Optional preferences: language, audience, tone, speaking rate.

## Output

Produce `NarrationPlan 1.0`, with content type, recommendation, timestamped narration segments, visual action, narration purpose, confidence, source evidence, estimated speech duration, fit status, no-voice windows and review notes.

See `references/narration-plan.schema.json` and `examples/narration-plan.json`.

## Scope boundary

Narrator generates text plans only. TTS is a separate stage. The resulting voiceover audio can then be passed to goldilock-av-mixer.
