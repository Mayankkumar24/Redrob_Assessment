"""
Step 1.4 — Disqualifier Flag Enrichment

Computes the "fuzzy" disqualifiers from the JD that need more than a single
field lookup. These are precomputed once across the full 100K pool so that
the runtime hard-filter step (Phase 2 / Step 2.2) is just a column lookup,
not recomputation.

NOTE on architecture vs. the original plan: the "production vs. research"
score originally planned for this step requires sentence embeddings (cosine
similarity against anchor sentences), which means it needs the embedding
model loaded. To avoid loading that model twice (once here, once in Step
1.5), that specific sub-check is computed in embed_candidates.py
immediately after candidate embeddings are generated, reusing the already-
loaded model. Everything else below is pure rule-based, no embeddings
needed, and lives here.

Three flags are computed:
  1. is_consulting_only        - entire career at consulting/services firms
  2. is_job_hopper             - avg tenure < 18 months across 3+ jobs
  3. cv_speech_only_flag       - CV/Speech/Robotics-heavy, NLP/IR-absent
"""

from __future__ import annotations

from typing import Any

from jd_requirements import load_jd_requirements


def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def check_consulting_only(
    career_history: list[dict], consulting_companies: list[str]
) -> tuple[bool, str]:
    """
    True only if EVERY company in career_history matches the consulting
    list. Per the JD: being CURRENTLY at a consulting firm is fine if prior
    companies show product-company experience - so a partial match does not
    trigger this flag, only a 100% match across the whole career.
    """
    if not career_history:
        return False, ""

    companies = [_normalize(j.get("company", "")) for j in career_history]
    matches = [
        any(cc in company for cc in consulting_companies) for company in companies
    ]

    if all(matches) and len(companies) > 0:
        return True, f"All {len(companies)} jobs at consulting/services firms"
    return False, ""


_SENIORITY_RANK = {
    "intern": 0, "junior": 1, "associate": 1, "engineer": 2, "analyst": 2,
    "senior": 3, "lead": 4, "staff": 5, "principal": 6, "director": 7,
    "vp": 8, "head": 7, "manager": 4, "architect": 5,
}


def _seniority_score(title: str) -> int:
    """Best-match seniority rank found in a title string. Default mid-level."""
    title = _normalize(title)
    matches = [rank for word, rank in _SENIORITY_RANK.items() if word in title]
    return max(matches) if matches else 2


def _is_title_escalating(career_history: list[dict]) -> bool:
    """
    True if seniority rank strictly increases (or stays flat at a high
    level) across the career in chronological order - the pattern the JD
    actually describes as "optimizing for Senior -> Staff -> Principal."
    Lateral moves (e.g. Search Engineer -> NLP Engineer -> Applied ML
    Engineer, all roughly the same level) do NOT count as escalating, even
    if tenure is short - that's domain-broadening, not title-chasing.
    """
    sorted_jobs = sorted(career_history, key=lambda j: j.get("start_date") or "")
    scores = [_seniority_score(j.get("title", "")) for j in sorted_jobs]
    increases = sum(1 for a, b in zip(scores, scores[1:]) if b > a)
    # Escalating if seniority increased at more than half of the transitions
    return len(scores) > 1 and increases >= (len(scores) - 1) / 2 and scores[-1] > scores[0]


def check_job_hopper(
    career_history: list[dict],
    tenure_threshold_months: int = 18,
    strict_tenure_threshold_months: int = 12,
    min_jobs: int = 3,
) -> tuple[bool, str]:
    """
    Flags the JD's actual concern: "switching companies every 1.5 years to
    chase Senior -> Staff -> Principal titles" - i.e. SHORT TENURE +
    ESCALATING TITLES together, not short tenure alone.

    Two paths to a flag:
      1. avg_tenure < tenure_threshold_months AND titles show an escalating
         seniority pattern (the literal JD pattern).
      2. avg_tenure < strict_tenure_threshold_months regardless of title
         pattern - even lateral moves this short are a reliability concern.

    A candidate with short-ish tenure but lateral/domain-broadening moves
    (like someone moving Recommendation -> Search -> NLP -> Applied ML at
    consistently strong companies) is NOT flagged unless tenure is very low,
    since that pattern reads as deliberate skill-building, not job-hopping.
    """
    if len(career_history) < min_jobs:
        return False, ""

    durations = [j.get("duration_months", 0) or 0 for j in career_history]
    avg_tenure = sum(durations) / len(durations)
    escalating = _is_title_escalating(career_history)

    if avg_tenure < strict_tenure_threshold_months:
        return True, (
            f"Avg tenure {avg_tenure:.1f}mo across {len(career_history)} jobs "
            f"is very short regardless of title pattern"
        )
    if avg_tenure < tenure_threshold_months and escalating:
        return True, (
            f"Avg tenure {avg_tenure:.1f}mo across {len(career_history)} jobs "
            f"WITH escalating seniority titles - matches JD's title-chaser pattern"
        )
    if avg_tenure < tenure_threshold_months and not escalating:
        return False, (
            f"Avg tenure {avg_tenure:.1f}mo is short but titles are lateral/"
            f"domain-broadening, not escalating - not flagged"
        )
    return False, ""


def check_cv_speech_only(
    skills: list[dict],
    career_history: list[dict],
    cv_speech_keywords: list[str],
    nlp_ir_keywords: list[str],
) -> tuple[bool, str]:
    """
    Flags candidates whose technical surface area is dominated by CV/Speech/
    Robotics terms with near-zero NLP/IR presence. Uses both skills[].name
    and career_history[].description text since skills alone can be sparse.
    """
    skill_text = " ".join(_normalize(s.get("name", "")) for s in skills)
    career_text = " ".join(_normalize(j.get("description", "")) for j in career_history)
    full_text = f"{skill_text} {career_text}"

    cv_speech_hits = sum(1 for kw in cv_speech_keywords if kw in full_text)
    nlp_ir_hits = sum(1 for kw in nlp_ir_keywords if kw in full_text)

    # Flag only when there's clear CV/Speech presence AND essentially no
    # NLP/IR presence - avoids over-flagging people who do both.
    if cv_speech_hits >= 2 and nlp_ir_hits == 0:
        return True, f"{cv_speech_hits} CV/Speech/Robotics terms, 0 NLP/IR terms"
    return False, ""


def compute_disqualifier_flags(
    records: list[dict[str, Any]], jd_requirements: dict | None = None
) -> dict[str, dict]:
    """
    Returns {candidate_id: {is_consulting_only, consulting_reason,
                             is_job_hopper, job_hopper_reason,
                             cv_speech_only_flag, cv_speech_reason}}
    """
    if jd_requirements is None:
        jd_requirements = load_jd_requirements()

    disq = jd_requirements["disqualifiers"]
    consulting_companies = disq["consulting_only_career"]["consulting_companies"]
    cv_speech_keywords = disq["cv_speech_robotics_without_nlp"]["cv_speech_robotics_keywords"]
    nlp_ir_keywords = disq["cv_speech_robotics_without_nlp"]["nlp_ir_keywords"]

    results = {}
    for r in records:
        cid = r["candidate_id"]
        career = r.get("career_history", [])
        skills = r.get("skills", [])

        is_consulting, consulting_reason = check_consulting_only(career, consulting_companies)
        is_hopper, hopper_reason = check_job_hopper(career)
        is_cv_speech, cv_speech_reason = check_cv_speech_only(
            skills, career, cv_speech_keywords, nlp_ir_keywords
        )

        results[cid] = {
            "is_consulting_only": is_consulting,
            "consulting_reason": consulting_reason,
            "is_job_hopper": is_hopper,
            "job_hopper_reason": hopper_reason,
            "cv_speech_only_flag": is_cv_speech,
            "cv_speech_reason": cv_speech_reason,
        }
    return results


if __name__ == "__main__":
    from io_utils import load_candidates

    records, _ = load_candidates(
        "data/raw/sample_candidates.json", "config/candidate_schema.json"
    )
    flags = compute_disqualifier_flags(records)

    print("=== Disqualifier flags across sample ===\n")
    n_consulting = sum(1 for f in flags.values() if f["is_consulting_only"])
    n_hopper = sum(1 for f in flags.values() if f["is_job_hopper"])
    n_cv_speech = sum(1 for f in flags.values() if f["cv_speech_only_flag"])
    print(f"Consulting-only: {n_consulting}/{len(flags)}")
    print(f"Job-hopper: {n_hopper}/{len(flags)}")
    print(f"CV/Speech-only (no NLP/IR): {n_cv_speech}/{len(flags)}")
    print()

    print("=== Sample flagged candidates ===")
    for cid, f in flags.items():
        reasons = []
        if f["is_consulting_only"]:
            reasons.append(f"CONSULTING: {f['consulting_reason']}")
        if f["is_job_hopper"]:
            reasons.append(f"HOPPER: {f['job_hopper_reason']}")
        if f["cv_speech_only_flag"]:
            reasons.append(f"CV/SPEECH: {f['cv_speech_reason']}")
        if reasons:
            print(f"{cid}: {' | '.join(reasons)}")
