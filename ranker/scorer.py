"""
ranker/scorer.py
─────────────────
Composite scoring pipeline.  Takes FAISS retrieval results +
candidate metadata and returns a scored, ranked DataFrame.

Score components
────────────────
1. semantic_score   – cosine similarity from FAISS (float in [0, 1])
2. skills_score     – hard-skill coverage against JD required/nice-to-have
3. experience_score – years-of-experience and seniority fit
4. behavioral_mult  – multiplier from redrob_signals (via behavioral.py)

Final formula
─────────────
    raw = W_sem * semantic
        + W_sk  * skills
        + W_exp * experience
        + W_beh * behavioral_component

    final = raw * behavioral_multiplier
    (multiplier boosts/penalises the raw score)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import polars as pl

from .behavioral import behavioral_multiplier, behavioral_breakdown
from .config import WEIGHTS


# ── Public entry point ────────────────────────────────────────────────────────

def score_candidates(
    faiss_ids:   np.ndarray,     # shape (K,) – indices into candidate_ids
    faiss_dists: np.ndarray,     # shape (K,) – cosine similarities (IP)
    candidate_ids: list[str],    # full ID list (parallel to FAISS)
    meta:        pl.DataFrame,
    jd_req:      dict[str, Any],
    disqualified: set[str],
) -> pl.DataFrame:
    """
    Score and rank the FAISS-retrieved candidates.

    Returns a Polars DataFrame sorted by final_score descending,
    with columns needed for CSV output and reasoning generation.
    """
    # Map FAISS index positions → candidate IDs + semantic scores
    cids   = [candidate_ids[i] for i in faiss_ids]
    sem_sc = faiss_dists.tolist()   # already normalised cosine similarities

    # Build a thin scoring frame for the retrieved set
    retrieved = pl.DataFrame({
        "candidate_id":   cids,
        "semantic_score": sem_sc,
    })

    # Join with full metadata
    scored = retrieved.join(meta, on="candidate_id", how="left")

    # Required JD fields for rule-based scoring
    req_skills     = {s.lower() for s in (jd_req.get("required_skills")      or [])}
    nice_skills    = {s.lower() for s in (jd_req.get("nice_to_have_skills")   or [])}
    min_exp        = jd_req.get("min_experience_years") or 0
    max_exp        = jd_req.get("max_experience_years") or 99
    seniority      = (jd_req.get("seniority_level") or "").lower()

    # Row-level scoring (Polars map_elements for complex nested logic)
    rows = scored.to_dicts()

    skills_scores  : list[float] = []
    exp_scores     : list[float] = []
    beh_mults      : list[float] = []
    final_scores   : list[float] = []
    beh_breakdowns : list[str]   = []

    for row in rows:
        sk  = _skills_score(row, req_skills, nice_skills)
        exp = _experience_score(row, min_exp, max_exp, seniority)
        bm  = behavioral_multiplier(row.get("redrob_signals") or {}, jd_req)
        bd  = behavioral_breakdown(row.get("redrob_signals") or {})

        raw = (
            WEIGHTS["semantic"]   * row["semantic_score"]
            + WEIGHTS["skills"]   * sk
            + WEIGHTS["experience"] * exp
            + WEIGHTS["behavioral"] * ((bm - 0.55) / (1.25 - 0.55))  # normalise mult to [0,1]
        )
        final = raw * bm   # multiplier amplifies / suppresses the raw score

        skills_scores.append(round(sk,  4))
        exp_scores.append(round(exp, 4))
        beh_mults.append(round(bm,   4))
        final_scores.append(round(final, 6))
        beh_breakdowns.append(_bd_str(bd))

    scored = scored.with_columns([
        pl.Series("skills_score",     skills_scores),
        pl.Series("experience_score", exp_scores),
        pl.Series("behavioral_mult",  beh_mults),
        pl.Series("behavioral_detail",beh_breakdowns),
        pl.Series("final_score",      final_scores),
        pl.Series("is_disqualified",  [c in disqualified for c in cids]),
    ])

    # Remove honeypots, sort descending
    scored = (
    scored
    .filter(pl.col("is_disqualified").not_())
    .sort(["final_score", "candidate_id"], descending=[True, False])
    )

    return scored


# ── Scoring sub-functions ─────────────────────────────────────────────────────

def _skills_score(
    row: dict,
    req_skills: set[str],
    nice_skills: set[str],
) -> float:
    """
    Fraction of required skills covered, with a bonus for nice-to-haves.

    skills_score = 0.80 * (required_covered / total_required)
                 + 0.20 * (nice_covered / total_nice)
    """
    candidate_skills: list[dict] = row.get("skills") or []
    cand_skill_names = {
        _normalise(s.get("name") or "") for s in candidate_skills
    }

    if req_skills:
        req_covered = sum(
            1 for rs in req_skills
            if any(_fuzzy_match(rs, cs) for cs in cand_skill_names)
        )
        req_score = req_covered / len(req_skills)
    else:
        req_score = 1.0

    if nice_skills:
        nice_covered = sum(
            1 for ns in nice_skills
            if any(_fuzzy_match(ns, cs) for cs in cand_skill_names)
        )
        nice_score = nice_covered / len(nice_skills)
    else:
        nice_score = 0.0

    return 0.80 * req_score + 0.20 * nice_score


def _experience_score(
    row: dict,
    min_exp: int,
    max_exp: int,
    target_seniority: str,
) -> float:
    """
    Combined years-of-experience fit + seniority level match.
    """
    total_exp: float = row.get("total_experience_years") or 0.0

    # Compute experience fit as a trapezoid:
    # 0 below min, linear ramp up to min, plateau until max, then slight drop
    if total_exp <= 0:
        exp_fit = 0.0
    elif total_exp < min_exp:
        exp_fit = total_exp / max(min_exp, 1)   # linear ramp
    elif total_exp <= max_exp:
        exp_fit = 1.0                            # plateau
    else:
        # Over-experienced: small penalty (too senior → likely to churn)
        overshot = total_exp - max_exp
        exp_fit = max(0.60, 1.0 - 0.03 * overshot)

    # Seniority level bonus
    cand_seniority = (row.get("seniority_level") or "").lower()
    seniority_bonus = 0.0

    if target_seniority and cand_seniority:
        if _seniority_match(cand_seniority, target_seniority):
            seniority_bonus = 0.20
        elif _seniority_adjacent(cand_seniority, target_seniority):
            seniority_bonus = 0.08

    return min(1.0, exp_fit * 0.80 + seniority_bonus)


# ── Helpers ───────────────────────────────────────────────────────────────────

_SENIORITY_LADDER = ["intern", "junior", "mid", "senior", "lead", "principal", "staff", "head", "vp", "cxo"]


def _seniority_match(cand: str, target: str) -> bool:
    for kw in _SENIORITY_LADDER:
        if kw in cand and kw in target:
            return True
    return False


def _seniority_adjacent(cand: str, target: str) -> bool:
    try:
        ci = next(i for i, kw in enumerate(_SENIORITY_LADDER) if kw in cand)
        ti = next(i for i, kw in enumerate(_SENIORITY_LADDER) if kw in target)
        return abs(ci - ti) == 1
    except StopIteration:
        return False


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _fuzzy_match(query: str, candidate_skill: str) -> bool:
    """Substring match on normalised strings (handles 'python3' vs 'python')."""
    q = _normalise(query)
    c = _normalise(candidate_skill)
    return q in c or c in q


def _bd_str(bd: dict[str, float]) -> str:
    """Compact breakdown string for the reasoning column."""
    return " | ".join(f"{k}={v:.2f}" for k, v in bd.items())
