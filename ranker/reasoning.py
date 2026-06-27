"""
ranker/reasoning.py
────────────────────
Generate 1-2 sentence reasoning strings for each top-100 candidate.

Requirements from submission_spec.md:
  • 1-2 sentences per candidate
  • Must explain WHY they rank highly (not generic praise)
  • Should reference specific skills, experience, and/or signals

Strategy
────────
Build reasoning from the highest-signal facts about each candidate:
  1. Top matching required skills (concrete)
  2. Years of experience vs JD requirement (concrete)
  3. Strongest behavioral signal bucket (concrete)
  4. One nice-to-have as a differentiator (if applicable)

Deliberately avoids vague phrases like "strong candidate" or "good fit".
"""

from __future__ import annotations

import re
from typing import Any


def generate_reasoning(
    row: dict[str, Any],
    jd_req: dict[str, Any],
) -> str:
    """
    Generate a 1-2 sentence reasoning string for a ranked candidate.

    Parameters
    ----------
    row : dict
        One row from the scored + ranked DataFrame (to_dicts()).
    jd_req : dict
        Parsed JD requirements.

    Returns
    -------
    str
        Reasoning string, max ~280 chars, exactly 1-2 sentences.
    """
    parts: list[str] = []

    # ── Sentence 1: Skills + Experience ──────────────────────────────────────
    s1_parts: list[str] = []

    matched_skills = _matched_skills(row, jd_req)
    if matched_skills:
        skill_str = ", ".join(matched_skills[:4])   # cap at 4 skills
        s1_parts.append(f"Covers {len(matched_skills)}/{_req_count(jd_req)} required skills including {skill_str}")

    exp_years = row.get("total_experience_years")
    min_exp   = jd_req.get("min_experience_years") or 0
    if exp_years is not None:
        exp_years_r = round(float(exp_years), 1)
        if exp_years_r >= min_exp:
            s1_parts.append(f"with {exp_years_r} years of relevant experience")
        else:
            s1_parts.append(f"with {exp_years_r} years of experience (below {min_exp}y requirement)")

    seniority = row.get("seniority_level")
    if seniority:
        s1_parts.append(f"at {seniority} level")

    if s1_parts:
        parts.append(_join_s1(s1_parts) + ".")

    # ── Sentence 2: Behavioral or differentiator ──────────────────────────────
    s2 = _behavioral_sentence(row, jd_req)
    if s2:
        parts.append(s2)

    if not parts:
        # Absolute fallback (should not happen in practice)
        parts.append(f"Ranked by composite semantic and skill alignment score of {row.get('final_score', 0):.3f}.")

    result = " ".join(parts)
    return _truncate(result, 300)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _matched_skills(row: dict, jd_req: dict) -> list[str]:
    """Return list of required skills the candidate demonstrably has."""
    req = [s.lower() for s in (jd_req.get("required_skills") or [])]
    candidate_skills: list[dict] = row.get("skills") or []
    cand_names = {_norm(s.get("name") or "") for s in candidate_skills}

    matched = []
    for rs in req:
        rs_norm = _norm(rs)
        if any(rs_norm in c or c in rs_norm for c in cand_names):
            # Use the original (nice casing) skill name from JD
            matched.append(_title_case(rs))
    return matched


def _req_count(jd_req: dict) -> int:
    return len(jd_req.get("required_skills") or [])


def _behavioral_sentence(row: dict, jd_req: dict) -> str:
    """
    Build a one-sentence behavioral summary using the strongest bucket.
    """
    bd_str: str = row.get("behavioral_detail") or ""
    mult: float = row.get("behavioral_mult") or 1.0

    if not bd_str:
        return ""

    # Parse the compact breakdown string: "consistency=0.82 | engagement=0.74 | …"
    buckets: dict[str, float] = {}
    for token in bd_str.split("|"):
        token = token.strip()
        if "=" in token:
            k, v = token.split("=", 1)
            try:
                buckets[k.strip()] = float(v)
            except ValueError:
                pass

    if not buckets:
        return ""

    best_bucket  = max(buckets, key=buckets.__getitem__)
    best_score   = buckets[best_bucket]
    worst_bucket = min(buckets, key=buckets.__getitem__)
    worst_score  = buckets[worst_bucket]

    # Multiplier descriptor
    if mult >= 1.15:
        mult_adj = "strong"
    elif mult >= 1.05:
        mult_adj = "solid"
    elif mult >= 0.95:
        mult_adj = "moderate"
    else:
        mult_adj = "below-average"

    bucket_labels = {
        "consistency": "profile consistency and tenure stability",
        "engagement":  "platform engagement and interview follow-through",
        "growth":      "skill progression velocity and continuous learning",
        "integrity":   "reference quality and salary expectation realism",
        "culture_fit": "cross-functional collaboration and culture alignment",
    }

    best_label = bucket_labels.get(best_bucket, best_bucket)

    s2 = f"Behavioral profile shows {mult_adj} signals ({mult:.2f}×), particularly in {best_label} ({best_score:.0%})"

    # Add a caveat if something is low
    if worst_score < 0.45 and worst_bucket != best_bucket:
        worst_label = bucket_labels.get(worst_bucket, worst_bucket)
        s2 += f"; weaker in {worst_label} ({worst_score:.0%})"

    return s2 + "."


def _join_s1(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0].capitalize()
    head = parts[0].capitalize()
    tail = " ".join(parts[1:])
    return f"{head} {tail}"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9+#.]", "", text.lower())


def _title_case(skill: str) -> str:
    """Smart title-case that preserves acronyms like AWS, SQL, CI/CD."""
    acronyms = {"aws", "sql", "gcp", "api", "ml", "ai", "llm", "nlp", "rag",
                "rest", "ci/cd", "k8s", "rds", "ec2", "s3", "etl", "elt",
                "dbt", "cka", "crm", "erp"}
    words = skill.split()
    out = []
    for w in words:
        out.append(w.upper() if w.lower() in acronyms else w.title())
    return " ".join(out)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    # Cut at last sentence boundary within limit
    cut = text[:max_len]
    last_period = cut.rfind(".")
    if last_period > max_len // 2:
        return cut[:last_period + 1]
    return cut.rstrip() + "…"
