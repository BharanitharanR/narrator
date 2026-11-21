"""Optional Gemini narration planner with model fallback."""
from __future__ import annotations

import json
import os
from typing import Any

from .fallback import model_chain, run_with_model_fallback
from .prompt import build_user_prompt, system_prompt


class GeminiNarrationProvider:
    """Calls Gemini once per model attempt. After create_plan, last_exchange holds
    {request, raw_text, model_used} for the caller to save (no API key in it)."""

    def __init__(self, model: str | None = None, api_key: str | None = None, client: Any = None):
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("Install narrator[gemini]") from exc
            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is required")
            client = genai.Client(api_key=key)
        self.client = client
        self.models = model_chain(model)
        self.last_exchange: dict[str, Any] | None = None

    def build_request(self, context: dict[str, Any], preferences: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_chain": self.models,
            "system_instruction": system_prompt(preferences.get("mode", "single")),
            "contents": build_user_prompt(context, preferences),
            "config": {"response_mime_type": "application/json", "temperature": 0.3},
        }

    def create_plan(self, *, context: dict[str, Any], preferences: dict[str, Any]) -> dict[str, Any]:
        request = self.build_request(context, preferences)
        self.last_exchange = {"request": request, "raw_text": None, "model_used": None}
        config = {"system_instruction": request["system_instruction"], **request["config"]}

        def attempt(model: str) -> str:
            response = self.client.models.generate_content(model=model, contents=request["contents"], config=config)
            body = getattr(response, "text", None)
            if not body:
                raise RuntimeError("Gemini returned empty output")
            return body

        body, used = run_with_model_fallback(self.models, attempt)
        self.last_exchange.update(raw_text=body, model_used=used)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Gemini returned JSON that is not an object")
        return data
