"""
ranker/scorer.py
─────────────────
Composite scoring — 3 base components + behavioral + skill penalty.

Formula:
  base_score  = 0.35 × prod_rescaled
              + 0.35 × semantic_score
              + 0.30 × skills_score

  adjusted    = base_score × behavioral_multiplier
  final_score = clamp(adjusted × (1 − skill_duration_penalty), 0, 1)

Three components capture different signals:
  prod_rescaled   → "production ML orientation" (Phase 1 anchor diff)
  semantic_score  → "fit for THIS specific JD"  (JD cosine sim)
  skills_score    → "has the exact required tools" (assessed + presence)
"""

from __future__ import annotations

import re
from typing import Any

from .behavioral import compute_behavioral_multiplier, behavioral_detail
from .config import (
    CURRENT_YEAR, TECH_INCEPTION_YEAR, SKILL_DURATION_PENALTY,
    WEIGHT_PRODUCTION, WEIGHT_SEMANTIC, WEIGHT_SKILLS,
)
from .skills import compute_skills_score, skills_breakdown


def score_candidate(
    cand_entry: dict[str, Any],
    row: dict[str, Any],
    semantic_score: float,
) -> dict[str, Any]:
    """
    Score a single candidate.

    Parameters
    ----------
    cand_entry     : {candidate_id, production_experience_score, prod_rescaled}
    row            : full row from candidates_clean.parquet
    semantic_score : cosine similarity with JD embedding, in [0, 1]

    Returns
    -------
    dict with all score components + _row for reasoning
    """
    cid           = cand_entry["candidate_id"]
    prod_rescaled = float(cand_entry.get("prod_rescaled", 0.5))
    sk_score      = compute_skills_score(row)

    # ── Base score (3 components) ─────────────────────────────────────────────
    base = (
        WEIGHT_PRODUCTION * prod_rescaled
        + WEIGHT_SEMANTIC  * semantic_score
        + WEIGHT_SKILLS    * sk_score
    )

    # ── Behavioral multiplier ─────────────────────────────────────────────────
    beh_mult = compute_behavioral_multiplier(row)

    # ── Skill duration penalty (magnitude-aware) ──────────────────────────────
    skill_pen, violations = _skill_duration_penalty(row)

    # ── Final score ───────────────────────────────────────────────────────────
    final_score = max(0.0, min(1.0, base * beh_mult * (1.0 - skill_pen)))

    return {
        "candidate_id":           cid,
        "prod_rescaled":          round(prod_rescaled, 6),
        "semantic_score":         round(semantic_score, 6),
        "skills_score":           round(sk_score, 6),
        "base_score":             round(base, 6),
        "behavioral_mult":        round(beh_mult, 6),
        "behavioral_detail":      behavioral_detail(row),
        "skill_duration_penalty": round(skill_pen, 6),
        "skill_violations":       violations,
        "skills_breakdown":       skills_breakdown(row),
        "final_score":            round(final_score, 6),
        "_row":                   row,
    }


# ── Magnitude-aware skill duration penalty ────────────────────────────────────

def _skill_duration_penalty(row: dict) -> tuple[float, list[dict]]:
    """
    overshoot_ratio = claimed_years / actual_max_years
    mild     ≤ 1.5  moderate ≤ 3.0  severe > 3.0
    penalty  = SKILL_DURATION_PENALTY[(min(count, 3), worst_severity)]
    """
    violations: list[dict] = []

    for skill in (row.get("skills") or []):
        if not isinstance(skill, dict):
            continue
        name  = _norm(skill.get("name") or "")
        years = skill.get("years_of_experience") or 0
        try:
            years = float(years)
        except (ValueError, TypeError):
            continue
        if years <= 0:
            continue

        for tech, inception_year in TECH_INCEPTION_YEAR.items():
            if tech in name:
                actual_max = CURRENT_YEAR - inception_year + 1
                if years > actual_max:
                    ratio    = years / max(actual_max, 1)
                    severity = (
                        "severe"   if ratio > 3.0 else
                        "moderate" if ratio > 1.5 else
                        "mild"
                    )
                    violations.append({
                        "skill":      skill.get("name"),
                        "claimed":    years,
                        "actual_max": actual_max,
                        "ratio":      round(ratio, 2),
                        "severity":   severity,
                    })
                break

    if not violations:
        return 0.0, []

    count_key = min(len(violations), 3)
    worst_sev = max(
        violations,
        key=lambda v: {"mild": 0, "moderate": 1, "severe": 2}.get(v["severity"], 0)
    )["severity"]
    penalty = SKILL_DURATION_PENALTY.get((count_key, worst_sev), 0.60)
    return penalty, violations


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()