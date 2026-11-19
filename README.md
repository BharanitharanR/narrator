# narrator

An Agent Skill for turning video understanding into timestamped AI voiceover scripts.

Narrator consumes [video-analysis](https://github.com/BharanitharanR/video-analysis) output. The analyser identifies video type; narrator uses that classification and semantic timeline to create evidence-grounded narration.

Pipeline: video -> video-analysis -> analysis.json -> narrator -> narration-plan.json -> TTS -> voiceover.wav -> goldilock-av-mixer -> final.mp4.

V1 generates concise narration segments with timestamps, evidence references, confidence, estimated speech duration, fit checks and no-voice windows. It does not perform TTS or audio mixing.

Narration is explanatory, not speculative. Each claim should be traceable to timestamped observations.
