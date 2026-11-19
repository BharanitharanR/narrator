"""Optional Gemini narration planner."""
from __future__ import annotations
import json,os
from typing import Any
from .prompt import SYSTEM_PROMPT,build_user_prompt

class GeminiNarrationProvider:
    def __init__(self,model:str="gemini-3.6-flash",api_key:str|None=None):
        try: from google import genai
        except ImportError as exc: raise RuntimeError("Install narrator[gemini]") from exc
        key=api_key or os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY is required")
        self.client=genai.Client(api_key=key); self.model=model
    def create_plan(self,*,context:dict[str,Any],preferences:dict[str,Any])->dict[str,Any]:
        response=self.client.models.generate_content(model=self.model,contents=build_user_prompt(context,preferences),config={"system_instruction":SYSTEM_PROMPT,"response_mime_type":"application/json","temperature":0.3})
        body=getattr(response,"text",None)
        if not body: raise RuntimeError("Gemini returned empty output")
        try: return json.loads(body)
        except json.JSONDecodeError as exc: raise ValueError("Gemini returned invalid JSON") from exc
