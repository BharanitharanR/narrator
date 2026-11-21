"""Narrator command line. One add_*/cmd_* pair per subcommand."""
from __future__ import annotations

import argparse
import json
import sys

from .planner import (
    build_planning_context,
    decide_for_analysis,
    load_analysis,
    plan_with_provider,
    write_plan,
)


def add_inspect_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("inspect", help="Show the video type, duration and narration decision")
    p.add_argument("analysis")
    p.set_defaults(func=cmd_inspect)


def cmd_inspect(args: argparse.Namespace) -> int:
    ctx = build_planning_context(load_analysis(args.analysis))
    decision = decide_for_analysis(args.analysis)
    print(json.dumps({"video_type": ctx["video_type"], "duration_seconds": ctx["duration_seconds"],
                      "screen_actions": len(ctx.get("screen_actions") or []),
                      "decision": {"narrate": decision["narrate"], "reason": decision["reason"]}}, indent=2))
    return 0


def _add_mode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["single", "dialogue"], default=None,
                   help="single (default) or dialogue; giving it explicitly also forces narration")
    p.add_argument("--force", action="store_true", help="Narrate even when the decision rule says no")


def add_decide_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("decide", help="Print the narrate-or-not decision as JSON (no API call)")
    p.add_argument("analysis")
    _add_mode_args(p)
    p.set_defaults(func=cmd_decide)


def cmd_decide(args: argparse.Namespace) -> int:
    print(json.dumps(decide_for_analysis(args.analysis, mode=args.mode, force=args.force), indent=2))
    return 0


def add_plan_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("plan", help="Write narration-plan.json (calls Gemini only when narrating)")
    p.add_argument("analysis")
    p.add_argument("-o", "--output", default="narration-plan.json")
    _add_mode_args(p)
    p.add_argument("--notes", default=None, help="Context from the person: product names, audience, emphasis")
    p.add_argument("--speech-segments", default=None, help="speech.json from goldilock-av-mixer detect-speech")
    p.add_argument("--speakers", default=None, help='"Name:role" (single) or "Name:role,Name:role" (dialogue)')
    p.add_argument("--voices", default=None, help="Gemini TTS voice names, e.g. Kore or Puck,Kore")
    p.add_argument("--model", default=None, help="First Gemini model to try (default: first of the chain)")
    p.add_argument("--language", default="en")
    p.add_argument("--audience", default="general")
    p.add_argument("--tone", default="friendly")
    p.add_argument("--words-per-minute", type=int, default=145)
    p.set_defaults(func=cmd_plan)


def _provider_factory(model: str | None):
    from .gemini import GeminiNarrationProvider
    return GeminiNarrationProvider(model)


class _LazyProvider:
    """Creates the Gemini client only if the planner actually calls it."""

    def __init__(self, model: str | None, factory=_provider_factory):
        self._model, self._factory, self._inner = model, factory, None
        self.last_exchange = None

    def create_plan(self, **kwargs):
        self._inner = self._factory(self._model)
        try:
            return self._inner.create_plan(**kwargs)
        finally:
            self.last_exchange = getattr(self._inner, "last_exchange", None)


def cmd_plan(args: argparse.Namespace, provider=None) -> int:
    provider = provider or _LazyProvider(args.model)
    plan = plan_with_provider(
        args.analysis, provider, mode=args.mode, force=args.force, speakers=args.speakers, voices=args.voices,
        speech_segments=args.speech_segments, output_path=args.output, notes=args.notes,
        language=args.language, audience=args.audience, tone=args.tone, target_words_per_minute=args.words_per_minute,
    )
    write_plan(plan, args.output)
    d = plan.decision
    if d is not None and not d.narrate:
        print(f"No narration: {d.reason}", file=sys.stderr)
    print(f"Narration plan written to {args.output}")
    return 0


def add_voice_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("voice", help="Voice a narration plan with Gemini TTS (voiceover.wav + timeline)")
    p.add_argument("plan", help="narration-plan.json")
    p.add_argument("-o", "--output-dir", default=".", help="Directory for voiceover.wav and voiceover-timeline.json")
    p.add_argument("--max-line-calls", type=int, default=12,
                   help="Cap on per-line TTS calls if the single take cannot be split")
    p.set_defaults(func=cmd_voice)


def cmd_voice(args: argparse.Namespace, client=None) -> int:
    from .voice import VoiceError, run_voice
    try:
        run_voice(args.plan, args.output_dir, client=client, max_line_calls=args.max_line_calls)
    except VoiceError as exc:
        print(f"narrator voice: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="narrator")
    sub = parser.add_subparsers(dest="command", required=True)
    add_inspect_parser(sub)
    add_decide_parser(sub)
    add_plan_parser(sub)
    add_voice_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
