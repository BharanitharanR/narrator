"""Regenerate references/narration-plan.schema.json from the pydantic model."""
import json
from pathlib import Path

from narrator.models import NarrationPlan


def schema() -> dict:
    s = NarrationPlan.model_json_schema()
    s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    s["title"] = "NarrationPlan 1.1"
    s["description"] = "Narration plan. 1.1 adds mode, speakers, decision and dialogue turns; 1.0 files still validate."
    return s


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "references" / "narration-plan.schema.json"
    out.write_text(json.dumps(schema(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
