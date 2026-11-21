"""Narrator command line."""
from __future__ import annotations
import argparse,json
from .planner import load_analysis,build_planning_context,plan_with_provider,write_plan
from .gemini import GeminiNarrationProvider

def main():
    parser=argparse.ArgumentParser(prog="narrator")
    sub=parser.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect"); inspect.add_argument("analysis")
    plan=sub.add_parser("plan"); plan.add_argument("analysis"); plan.add_argument("-o","--output",default="narration-plan.json"); plan.add_argument("--model",default="gemini-3.6-flash"); plan.add_argument("--language",default="en"); plan.add_argument("--audience",default="general"); plan.add_argument("--tone",default="friendly"); plan.add_argument("--words-per-minute",type=int,default=145)
    _add_voice(sub)
    args=parser.parse_args()
    if hasattr(args,"func"): raise SystemExit(args.func(args))
    if args.command=="inspect":
        ctx=build_planning_context(load_analysis(args.analysis)); kind=ctx["video_type"].get("primary","unknown")
        suitable=kind in {"demo","tutorial","walkthrough","presentation","documentation","screen_recording","review","news","meeting"}
        print(json.dumps({"video_type":ctx["video_type"],"duration_seconds":ctx["duration_seconds"],"narration_suitability":"recommended" if suitable else "optional_or_review"},indent=2)); return
    provider=GeminiNarrationProvider(args.model)
    result=plan_with_provider(args.analysis,provider,language=args.language,audience=args.audience,tone=args.tone,target_words_per_minute=args.words_per_minute)
    write_plan(result,args.output); print(f"Narration plan written to {args.output}")

def _add_voice(sub):
    """`narrator voice` subcommand (voiceover stage, see voice.py)."""
    from .voice import add_voice_command
    add_voice_command(sub)

if __name__=="__main__": main()
