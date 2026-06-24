"""
Step 1.1 (continued) — Flattening

Converts the nested candidate records (profile / career_history / education /
skills / redrob_signals) into ONE flat Polars DataFrame with scalar columns,
so that everything downstream (filtering, scoring) can be fast vectorized
Polars operations instead of slow row-by-row Python loops.

Design choice: nested arrays (career_history, education, skills) are
*summarized* into scalar/list columns here (e.g. num_jobs, avg_tenure_months,
companies list). The full raw nested record is kept separately (the
`records` list of dicts) for any step that genuinely needs row-by-row nested
access — e.g. honeypot date-math checks. This hybrid avoids fighting
Polars' Struct/List type system for things that don't need it, while still
getting Polars' speed for the 100K-row filtering/scoring that matters most.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl


def _safe_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _career_stats(career_history: list[dict]) -> dict:
    """Summarize the career_history array into scalar/list features."""
    durations = [j.get("duration_months", 0) or 0 for j in career_history]
    companies = [j.get("company", "") for j in career_history]
    industries = [j.get("industry", "") for j in career_history]
    titles = [j.get("title", "") for j in career_history]
    company_sizes = [j.get("company_size", "") for j in career_history]

    return {
        "num_jobs": len(career_history),
        "avg_tenure_months": round(sum(durations) / len(durations), 1) if durations else 0.0,
        "min_tenure_months": min(durations) if durations else 0,
        "max_tenure_months": max(durations) if durations else 0,
        "total_career_months": sum(durations),
        "companies": companies,
        "industries": industries,
        "titles_history": titles,
        "company_sizes_history": company_sizes,
    }


_TIER_RANK = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}


def _education_stats(education: list[dict]) -> dict:
    if not education:
        return {
            "num_education": 0,
            "best_education_tier": "unknown",
            "degrees": [],
            "fields_of_study": [],
        }
    tiers = [e.get("tier", "unknown") for e in education]
    best_tier = min(tiers, key=lambda t: _TIER_RANK.get(t, 5))
    return {
        "num_education": len(education),
        "best_education_tier": best_tier,
        "degrees": [e.get("degree", "") for e in education],
        "fields_of_study": [e.get("field_of_study", "") for e in education],
    }


def _skill_stats(skills: list[dict]) -> dict:
    names = [s.get("name", "") for s in skills]
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    return {
        "num_skills": len(skills),
        "num_expert_skills": expert_count,
        "skill_names": names,
        "skill_names_lower": [n.lower() for n in names],
    }


def _signal_stats(sig: dict) -> dict:
    assessments = sig.get("skill_assessment_scores", {}) or {}
    gh = sig.get("github_activity_score", -1)
    offer_rate = sig.get("offer_acceptance_rate", -1)
    salary = sig.get("expected_salary_range_inr_lpa", {}) or {}

    return {
        "redrob_profile_completeness_score": sig.get("profile_completeness_score"),
        "redrob_signup_date": sig.get("signup_date"),
        "redrob_last_active_date": sig.get("last_active_date"),
        "redrob_open_to_work_flag": sig.get("open_to_work_flag"),
        "redrob_profile_views_received_30d": sig.get("profile_views_received_30d"),
        "redrob_applications_submitted_30d": sig.get("applications_submitted_30d"),
        "redrob_recruiter_response_rate": sig.get("recruiter_response_rate"),
        "redrob_avg_response_time_hours": sig.get("avg_response_time_hours"),
        "redrob_num_assessments": len(assessments),
        "redrob_avg_assessment_score": (
            round(sum(assessments.values()) / len(assessments), 1) if assessments else None
        ),
        "redrob_connection_count": sig.get("connection_count"),
        "redrob_endorsements_received": sig.get("endorsements_received"),
        "redrob_notice_period_days": sig.get("notice_period_days"),
        "redrob_expected_salary_min": salary.get("min"),
        "redrob_expected_salary_max": salary.get("max"),
        "redrob_preferred_work_mode": sig.get("preferred_work_mode"),
        "redrob_willing_to_relocate": sig.get("willing_to_relocate"),
        "redrob_github_activity_score": gh,
        "redrob_has_github": gh is not None and gh != -1,
        "redrob_search_appearance_30d": sig.get("search_appearance_30d"),
        "redrob_saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d"),
        "redrob_interview_completion_rate": sig.get("interview_completion_rate"),
        "redrob_offer_acceptance_rate": offer_rate,
        "redrob_has_offer_history": offer_rate is not None and offer_rate != -1,
        "redrob_verified_email": sig.get("verified_email"),
        "redrob_verified_phone": sig.get("verified_phone"),
        "redrob_linkedin_connected": sig.get("linkedin_connected"),
    }


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one nested candidate dict into one flat row dict."""
    profile = record.get("profile", {})
    flat = {
        "candidate_id": record["candidate_id"],
        "anonymized_name": profile.get("anonymized_name"),
        "headline": profile.get("headline"),
        "summary": profile.get("summary"),
        "location": profile.get("location"),
        "country": profile.get("country"),
        "years_of_experience": profile.get("years_of_experience"),
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "current_company_size": profile.get("current_company_size"),
        "current_industry": profile.get("current_industry"),
    }
    flat.update(_career_stats(record.get("career_history", [])))
    flat.update(_education_stats(record.get("education", [])))
    flat.update(_skill_stats(record.get("skills", [])))
    flat.update(_signal_stats(record.get("redrob_signals", {})))
    return flat


def build_flat_dataframe(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Flatten all records and build the Polars DataFrame."""
    rows = [flatten_record(r) for r in records]
    df = pl.DataFrame(rows, infer_schema_length=None)
    return df


if __name__ == "__main__":
    import json
    from io_utils import load_candidates

    records, _ = load_candidates(
        "data/raw/sample_candidates.json", "config/candidate_schema.json"
    )
    df = build_flat_dataframe(records)
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns}")
    print()
    print(df.select(
        "candidate_id", "current_title", "years_of_experience",
        "num_jobs", "avg_tenure_months", "redrob_has_github",
        "redrob_recruiter_response_rate"
    ).head(10))
