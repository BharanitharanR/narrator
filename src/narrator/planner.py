"""Read analyser artifacts and orchestrate narration planning."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .models import NarrationPlan

def load_analysis(path: str | Path) -> dict[str,Any]:
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data,dict) or "video" not in data: raise ValueError("Expected video-analysis output")
    return data

def classification_from_analysis(analysis:dict[str,Any])->dict[str,Any]:
    dims=analysis.get("video",{}).get("dimensions",{})
    return dims.get("video_type") or {"primary":"unknown","confidence":0.0,"alternatives":[],"reason":"No video_type classification in source analysis"}

def build_planning_context(analysis:dict[str,Any])->dict[str,Any]:
    dims=analysis.get("video",{}).get("dimensions",{})
    sem=dims.get("semantic",{})
    tech=dims.get("technical",{})
    return {"duration_seconds":float(tech.get("duration_seconds") or sem.get("duration_seconds") or 0),"video_type":classification_from_analysis(analysis),"semantic":sem,"scenes":dims.get("scenes",{}),"screen":dims.get("screen",{})}

def plan_with_provider(analysis_path:str|Path,provider:Any,**preferences:Any)->NarrationPlan:
    context=build_planning_context(load_analysis(analysis_path))
    if context["duration_seconds"]<=0: raise ValueError("No valid source duration")
    kind=context["video_type"].get("primary","unknown")
    if kind=="unknown": raise ValueError("Video type unknown; run semantic classification first")
    raw=provider.create_plan(context=context,preferences=preferences)
    raw.setdefault("video_type",kind)
    raw.setdefault("classification_confidence",context["video_type"].get("confidence",0))
    raw.setdefault("source_analysis",str(analysis_path))
    raw.setdefault("duration_seconds",context["duration_seconds"])
    plan=NarrationPlan.model_validate(raw)
    if abs(plan.duration_seconds-context["duration_seconds"])>0.25: raise ValueError("Narration duration must match source video")
    return plan

def write_plan(plan:NarrationPlan,output_path:str|Path)->None:
    Path(output_path).write_text(json.dumps(plan.model_dump(mode="json"),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
