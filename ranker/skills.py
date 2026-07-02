"""
ranker/skills.py
─────────────────
JD skill matching logic.

Reads candidate's skill_assessment_scores (from redrob_signals)
and their skills list (from profile), normalises to lowercase,
then matches against MUST_HAVE_SKILLS and NICE_TO_HAVE_SKILLS
category dicts defined in config.py.

skills_score = 0.80 × must_have_avg + 0.20 × nice_have_avg

Where:
  must_have_avg = average assessment score (0-1) across all 4
                  must-have categories (unmatched category = 0)
  nice_have_avg = average assessment score across 5 nice-to-have
                  categories (unmatched = 0)
"""

from __future__ import annotations
import re
from typing import Any

from .config import MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, WEIGHT_MUST_HAVE, WEIGHT_NICE_HAVE


def compute_skills_score(row: dict[str, Any]) -> float:
    """
    Returns skills_score in [0, 1].
    """
    # Pull skill_assessment_scores from redrob_signals
    signals = _get_signals(row)
    assessment_scores: dict[str, float] = {}

    raw_assessments = signals.get("skill_assessment_scores") or {}
    if isinstance(raw_assessments, dict):
        for skill_name, score in raw_assessments.items():
            normalised = _normalise(skill_name)
            try:
                assessment_scores[normalised] = float(score) / 100.0  # → [0, 1]
            except (ValueError, TypeError):
                pass

    # Also build a set of all skills from the candidate profile (names only)
    # This is used as a binary presence signal when no assessment score exists
    profile_skills: set[str] = set()
    for skill_entry in (row.get("skills") or []):
        if isinstance(skill_entry, dict):
            name = skill_entry.get("name") or ""
        else:
            name = str(skill_entry)
        profile_skills.add(_normalise(name))

    # ── Must-have scoring ─────────────────────────────────────────────────────
    must_scores: list[float] = []
    for category, aliases in MUST_HAVE_SKILLS.items():
        score = _best_match_score(aliases, assessment_scores, profile_skills)
        must_scores.append(score)

    must_avg = sum(must_scores) / len(must_scores) if must_scores else 0.0

    # ── Nice-to-have scoring ──────────────────────────────────────────────────
    nice_scores: list[float] = []
    for category, aliases in NICE_TO_HAVE_SKILLS.items():
        score = _best_match_score(aliases, assessment_scores, profile_skills)
        nice_scores.append(score)

    nice_avg = sum(nice_scores) / len(nice_scores) if nice_scores else 0.0

    return round(WEIGHT_MUST_HAVE * must_avg + WEIGHT_NICE_HAVE * nice_avg, 6)


def skills_breakdown(row: dict[str, Any]) -> dict[str, float]:
    """
    Returns per-category scores for reasoning generation.
    Format: {"embeddings": 0.77, "vector_database": 0.53, ...}
    """
    signals = _get_signals(row)
    assessment_scores: dict[str, float] = {}
    raw = signals.get("skill_assessment_scores") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                assessment_scores[_normalise(k)] = float(v) / 100.0
            except (ValueError, TypeError):
                pass

    profile_skills: set[str] = set()
    for s in (row.get("skills") or []):
        name = s.get("name") or "" if isinstance(s, dict) else str(s)
        profile_skills.add(_normalise(name))

    result = {}
    for category, aliases in MUST_HAVE_SKILLS.items():
        result[category] = round(_best_match_score(aliases, assessment_scores, profile_skills), 3)
    for category, aliases in NICE_TO_HAVE_SKILLS.items():
        result[category] = round(_best_match_score(aliases, assessment_scores, profile_skills), 3)
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _best_match_score(
    aliases: list[str],
    assessment_scores: dict[str, float],
    profile_skills: set[str],
) -> float:
    """
    For a given skill category (list of aliases):
    1. Find best assessment score among all matching aliases
    2. If no assessment score but skill present in profile → give 0.50 (presence bonus)
    3. If not found at all → 0.0
    """
    best_assessment = None

    for alias in aliases:
        alias_norm = _normalise(alias)
        # Exact match first
        if alias_norm in assessment_scores:
            val = assessment_scores[alias_norm]
            if best_assessment is None or val > best_assessment:
                best_assessment = val
            continue
        # Substring match in assessment keys
        for key, val in assessment_scores.items():
            if alias_norm in key or key in alias_norm:
                if best_assessment is None or val > best_assessment:
                    best_assessment = val

    if best_assessment is not None:
        return best_assessment

    # No assessment score — check profile skill names (binary presence)
    for alias in aliases:
        alias_norm = _normalise(alias)
        for ps in profile_skills:
            if alias_norm in ps or ps in alias_norm:
                return 0.50   # presence bonus (no verified score)

    return 0.0


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation for comparison."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _get_signals(row: dict) -> dict:
    """Extract redrob_signals safely — handles nested dict or flat columns."""
    signals = row.get("redrob_signals")
    if isinstance(signals, dict):
        return signals
    # Might be stored as JSON string (some parquet writers do this)
    if isinstance(signals, str):
        import json
        try:
            return json.loads(signals)
        except Exception:
            pass
    return {}
