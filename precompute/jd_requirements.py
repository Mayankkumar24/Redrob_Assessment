"""
Step 1.3 — JD Understanding

The JD doesn't change between runs, so it is decomposed into structured
requirements EXACTLY ONCE, offline, and the result is hardcoded into
config/jd_requirements.json (already checked into the repo).

Runtime (Phase 2 / rank.py) NEVER calls an LLM for this — it just loads the
static JSON below. This keeps the timed ranking run fully network-free.

This module provides two things:
  1. load_jd_requirements()      - the function the rest of the pipeline uses
  2. regenerate_with_gemini()    - OPTIONAL helper if the JD changes and you
                                    want to re-extract requirements using
                                    Gemini 1.5 Pro instead of hand-editing
                                    the JSON. Requires GEMINI_API_KEY env var
                                    and network access. Not used by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_jd_requirements(path: str = "config/jd_requirements.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jd_text(path: str = "data/raw/job_description.md") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


GEMINI_EXTRACTION_PROMPT = """You are analyzing a job description for a technical recruiting AI system.
Read the job description below CAREFULLY, including everything between the lines,
not just the literal skills list. Pay special attention to any section that
describes what the role does NOT want, what the "ideal candidate" looks like,
and any explicit disqualifiers.

Return ONLY a JSON object (no markdown fences, no preamble) matching this exact
structure:

{
  "role_title": "...",
  "experience_range": {"min_years": 0, "max_years": 0, "ideal_min_years": 0, "ideal_max_years": 0},
  "hard_requirements": [{"id": "...", "description": "...", "keywords": ["..."]}],
  "nice_to_have": [{"id": "...", "keywords": ["..."]}],
  "disqualifiers": {"<id>": {"description": "...", "severity": "hard|soft", "detection_hint": "..."}},
  "location": {"preferred_cities": [], "acceptable_cities": [], "country_required": "...", "work_mode": "..."},
  "notice_period": {"ideal_max_days": 0},
  "ideal_candidate_summary": {"total_experience_years": "...", "company_type": "...", "has_shipped": "..."}
}

JOB DESCRIPTION:
---
{jd_text}
---
"""


def regenerate_with_gemini(
    jd_path: str = "data/raw/job_description.md",
    output_path: str = "config/jd_requirements_gemini.json",
    model_name: str = "gemini-1.5-pro",
) -> dict:
    """
    OPTIONAL utility. Calls Gemini 1.5 Pro to re-extract JD requirements.
    Not called by the main pipeline (run_pipeline.py) by default, since the
    hand-curated config/jd_requirements.json already reflects a careful,
    line-by-line read of the JD including its hackathon-specific traps.

    Use this only if the JD changes and you want a fresh first-pass
    extraction to review/edit, rather than starting from scratch.

    Requires: pip install google-generativeai
              export GEMINI_API_KEY=...
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. This function is optional and not "
            "required for the pipeline to run - config/jd_requirements.json "
            "is already hand-curated and checked into the repo."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    jd_text = load_jd_text(jd_path)
    prompt = GEMINI_EXTRACTION_PROMPT.replace("{jd_text}", jd_text)

    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Strip markdown fences if the model added them despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    parsed = json.loads(raw_text)

    Path(output_path).write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"[jd_requirements] Gemini extraction saved to {output_path}")
    print("Review it against config/jd_requirements.json before using it to "
          "replace the hand-curated version.")
    return parsed


if __name__ == "__main__":
    reqs = load_jd_requirements()
    print("Loaded JD requirements:")
    print(f"  Role: {reqs['role_title']}")
    print(f"  Experience range: {reqs['experience_range']}")
    print(f"  Hard requirements: {[r['id'] for r in reqs['hard_requirements']]}")
    print(f"  Disqualifiers: {list(reqs['disqualifiers'].keys())}")
    print(f"  Preferred cities: {reqs['location']['preferred_cities']}")
