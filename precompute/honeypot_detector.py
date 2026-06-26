# """
# Step 1.7 — Honeypot Detection

# Pre-flags the ~80 honeypots (subtly impossible profiles) described in
# redrob_signals_doc.md, so that Phase 2's hard-filter step (2.2) is just a
# column lookup, not recomputation across 100K rows.

# KEY DESIGN DECISION — two tiers, not one:

#   HARD flags (honeypot_flag = True -> excluded entirely in Step 2.2):
#     Things that are logically/statistically IMPOSSIBLE, matching the doc's
#     own language ("subtly impossible profiles"). E.g. claiming "expert"
#     proficiency in a skill used for 0 months, or education ending before
#     it started. These cannot be true of a real person.

#   SOFT flags (contradiction_flags, contradiction_score -> used as a
#   down-weighting multiplier in Phase 2 scoring, NOT auto-excluded):
#     Things that are unusual/suspicious but not actually impossible. E.g.
#     open_to_work=False while submitting 9 applications in 30 days IS
#     plausible in real life (someone quietly testing the market without
#     publicly flagging it to their current employer). Auto-dropping these
#     risks false-positive-excluding genuinely good candidates and hurting
#     NDCG@10 - the spec's dominant scoring metric. These get penalized via
#     scoring instead of removed outright.

# This split is deliberate and defensible in a Stage 5 interview: "honeypot"
# in the spec means impossible, not merely unusual.
# """

# from __future__ import annotations

# from datetime import date
# from typing import Any


# def _parse_date(s: str | None) -> date | None:
#     if not s:
#         return None
#     try:
#         return date.fromisoformat(s)
#     except (ValueError, TypeError):
#         return None


# def _months_between(d1: date, d2: date) -> int:
#     return (d2.year - d1.year) * 12 + (d2.month - d1.month)


# # ---------------------------------------------------------------------------
# # HARD checks — logical/statistical impossibilities
# # ---------------------------------------------------------------------------

# def _check_expert_zero_duration(skills: list[dict]) -> list[str]:
#     reasons = []
#     for s in skills:
#         if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0:
#             reasons.append(f"expert proficiency in '{s.get('name')}' with 0 months used")
#     return reasons


# def _check_all_assessments_perfect(redrob_signals: dict) -> list[str]:
#     assessments = redrob_signals.get("skill_assessment_scores", {}) or {}
#     if len(assessments) >= 2 and all(v == 100 for v in assessments.values()):
#         return [f"all {len(assessments)} skill assessment scores are exactly 100"]
#     return []


# def _check_education_date_impossible(education: list[dict]) -> list[str]:
#     reasons = []
#     for e in education:
#         start, end = e.get("start_year"), e.get("end_year")
#         if start is not None and end is not None and end < start:
#             reasons.append(f"education end_year ({end}) before start_year ({start}) at {e.get('institution')}")
#     return reasons


# def _check_career_date_math(career_history: list[dict]) -> list[str]:
#     """duration_months must roughly match (end_date - start_date), and a
#     current job (is_current=True) should have end_date=null."""
#     reasons = []
#     for j in career_history:
#         start = _parse_date(j.get("start_date"))
#         end = _parse_date(j.get("end_date"))
#         claimed = j.get("duration_months", 0) or 0

#         if j.get("is_current") and end is not None:
#             reasons.append(f"is_current=True but end_date is set ({j.get('end_date')}) at {j.get('company')}")

#         if start and end:
#             actual = _months_between(start, end)
#             if abs(actual - claimed) > 3:  # small tolerance for day-of-month rounding
#                 reasons.append(
#                     f"duration_months ({claimed}) doesn't match start/end dates "
#                     f"(~{actual}mo actual) at {j.get('company')}"
#                 )
#         if start and end and end < start:
#             reasons.append(f"end_date before start_date at {j.get('company')}")
#     return reasons


# def _check_all_reachability_perfect(redrob_signals: dict) -> list[str]:
#     """recruiter_response_rate, interview_completion_rate, and
#     offer_acceptance_rate ALL simultaneously at the theoretical maximum
#     (1.0) is the kind of "too good to be true" combination the doc
#     describes - a real person's behavior has natural variance."""
#     rr = redrob_signals.get("recruiter_response_rate")
#     icr = redrob_signals.get("interview_completion_rate")
#     oar = redrob_signals.get("offer_acceptance_rate")
#     if rr == 1.0 and icr == 1.0 and oar == 1.0:
#         return ["recruiter_response_rate, interview_completion_rate, and "
#                 "offer_acceptance_rate are ALL simultaneously 1.0"]
#     return []


# def _check_experience_vs_career_history(profile: dict, career_history: list[dict]) -> list[str]:
#     """years_of_experience should be roughly >= the span implied by career
#     history (allowing gaps, so we only flag a LARGE mismatch, not a small one)."""
#     yoe = profile.get("years_of_experience")
#     total_months = sum(j.get("duration_months", 0) or 0 for j in career_history)
#     total_years = total_months / 12
#     if yoe is not None and total_years > yoe + 2:  # 2yr tolerance for overlap/rounding
#         return [f"career_history implies ~{total_years:.1f}yrs but "
#                 f"years_of_experience claims only {yoe}"]
#     return []


# HARD_CHECKS = [
#     ("expert_zero_duration", lambda r: _check_expert_zero_duration(r.get("skills", []))),
#     ("all_assessments_perfect", lambda r: _check_all_assessments_perfect(r.get("redrob_signals", {}))),
#     ("education_date_impossible", lambda r: _check_education_date_impossible(r.get("education", []))),
#     ("career_date_math", lambda r: _check_career_date_math(r.get("career_history", []))),
#     ("all_reachability_perfect", lambda r: _check_all_reachability_perfect(r.get("redrob_signals", {}))),
#     ("experience_mismatch", lambda r: _check_experience_vs_career_history(
#         r.get("profile", {}), r.get("career_history", []))),
# ]


# # ---------------------------------------------------------------------------
# # SOFT checks — suspicious but not impossible; down-weight, don't exclude
# # ---------------------------------------------------------------------------

# def _check_availability_contradiction(redrob_signals: dict) -> list[str]:
#     if (
#         redrob_signals.get("open_to_work_flag") is False
#         and (redrob_signals.get("applications_submitted_30d") or 0) > 8
#     ):
#         return [f"open_to_work=False but submitted "
#                 f"{redrob_signals.get('applications_submitted_30d')} applications in 30d"]
#     return []


# def _check_salary_anomaly(profile: dict, redrob_signals: dict) -> list[str]:
#     yoe = profile.get("years_of_experience") or 0
#     salary = redrob_signals.get("expected_salary_range_inr_lpa", {}) or {}
#     max_sal = salary.get("max")
#     if yoe >= 8 and max_sal is not None and max_sal < 10:
#         return [f"{yoe}yrs experience but expects max {max_sal} LPA "
#                 f"(unusually low for seniority)"]
#     return []


# SOFT_CHECKS = [
#     ("availability_contradiction", lambda r: _check_availability_contradiction(r.get("redrob_signals", {}))),
#     ("salary_anomaly", lambda r: _check_salary_anomaly(r.get("profile", {}), r.get("redrob_signals", {}))),
# ]


# def evaluate_candidate(record: dict[str, Any]) -> dict:
#     hard_reasons = []
#     for _, check_fn in HARD_CHECKS:
#         hard_reasons.extend(check_fn(record))

#     soft_reasons = []
#     for _, check_fn in SOFT_CHECKS:
#         soft_reasons.extend(check_fn(record))

#     return {
#         "candidate_id": record["candidate_id"],
#         "honeypot_flag": len(hard_reasons) > 0,
#         "honeypot_reasons": hard_reasons,
#         "contradiction_flag": len(soft_reasons) > 0,
#         "contradiction_reasons": soft_reasons,
#     }


# def detect_honeypots(records: list[dict[str, Any]]) -> dict[str, dict]:
#     return {r["candidate_id"]: evaluate_candidate(r) for r in records}


# if __name__ == "__main__":
#     from io_utils import load_candidates

#     records, _ = load_candidates(
#         "data/raw/sample_candidates.json", "config/candidate_schema.json"
#     )
#     results = detect_honeypots(records)

#     n_hard = sum(1 for r in results.values() if r["honeypot_flag"])
#     n_soft = sum(1 for r in results.values() if r["contradiction_flag"])
#     print(f"=== Honeypot detection across {len(results)} sample candidates ===\n")
#     print(f"HARD honeypot flags (would be excluded): {n_hard}/{len(results)} "
#           f"({100*n_hard/len(results):.1f}%)")
#     print(f"SOFT contradiction flags (down-weighted only): {n_soft}/{len(results)} "
#           f"({100*n_soft/len(results):.1f}%)")
#     print()

#     if n_hard:
#         print("--- HARD flags (detail) ---")
#         for cid, r in results.items():
#             if r["honeypot_flag"]:
#                 print(f"{cid}:")
#                 for reason in r["honeypot_reasons"]:
#                     print(f"  - {reason}")

#     print("\n--- SOFT flags (detail) ---")
#     for cid, r in results.items():
#         if r["contradiction_flag"]:
#             print(f"{cid}:")
#             for reason in r["contradiction_reasons"]:
#                 print(f"  - {reason}")

#     print(f"\nNote: spec disqualification threshold is >10% HARD honeypots in "
#           f"the TOP 100 of your final submission, not across the whole pool. "
#           f"This sample has 0 designed honeypots by chance (the full pool has "
#           f"~80 out of 100,000 = 0.08%, so a 50-candidate random sample "
#           f"often won't contain a clean one) - this script's hard-check logic "
#           f"is what matters; verify it correctly against true honeypots once "
#           f"you have the full candidates.jsonl.gz file.")


"""
Step 1.7 — Honeypot Detection

Pre-flags the ~80 honeypots (subtly impossible profiles) described in
redrob_signals_doc.md, so that Phase 2's hard-filter step (2.2) is just a
column lookup, not recomputation across 100K rows.

KEY DESIGN DECISION — two tiers, not one:

  HARD flags (honeypot_flag = True -> excluded entirely in Step 2.2):
    Things that are logically/statistically IMPOSSIBLE, matching the doc's
    own language ("subtly impossible profiles"). E.g. claiming "expert"
    proficiency in a skill used for 0 months, or education ending before
    it started. These cannot be true of a real person.

  SOFT flags (contradiction_flags, contradiction_score -> used as a
  down-weighting multiplier in Phase 2 scoring, NOT auto-excluded):
    Things that are unusual/suspicious but not actually impossible. E.g.
    open_to_work=False while submitting 9 applications in 30 days IS
    plausible in real life (someone quietly testing the market without
    publicly flagging it to their current employer). Auto-dropping these
    risks false-positive-excluding genuinely good candidates and hurting
    NDCG@10 - the spec's dominant scoring metric. These get penalized via
    scoring instead of removed outright.

This split is deliberate and defensible in a Stage 5 interview: "honeypot"
in the spec means impossible, not merely unusual.
"""

from __future__ import annotations

from datetime import date
from typing import Any


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _months_between(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


# ---------------------------------------------------------------------------
# HARD checks — logical/statistical impossibilities
# ---------------------------------------------------------------------------

def _check_expert_zero_duration(skills: list[dict]) -> list[str]:
    reasons = []
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0:
            reasons.append(f"expert proficiency in '{s.get('name')}' with 0 months used")
    return reasons


def _check_all_assessments_perfect(redrob_signals: dict) -> list[str]:
    assessments = redrob_signals.get("skill_assessment_scores", {}) or {}
    if len(assessments) >= 2 and all(v == 100 for v in assessments.values()):
        return [f"all {len(assessments)} skill assessment scores are exactly 100"]
    return []


def _check_education_date_impossible(education: list[dict]) -> list[str]:
    reasons = []
    for e in education:
        start, end = e.get("start_year"), e.get("end_year")
        if start is not None and end is not None and end < start:
            reasons.append(f"education end_year ({end}) before start_year ({start}) at {e.get('institution')}")
    return reasons


def _check_career_date_math(career_history: list[dict]) -> list[str]:
    """duration_months must roughly match (end_date - start_date), and a
    current job (is_current=True) should have end_date=null."""
    reasons = []
    for j in career_history:
        start = _parse_date(j.get("start_date"))
        end = _parse_date(j.get("end_date"))
        claimed = j.get("duration_months", 0) or 0

        if j.get("is_current") and end is not None:
            reasons.append(f"is_current=True but end_date is set ({j.get('end_date')}) at {j.get('company')}")

        if start and end:
            actual = _months_between(start, end)
            if abs(actual - claimed) > 3:  # small tolerance for day-of-month rounding
                reasons.append(
                    f"duration_months ({claimed}) doesn't match start/end dates "
                    f"(~{actual}mo actual) at {j.get('company')}"
                )
        if start and end and end < start:
            reasons.append(f"end_date before start_date at {j.get('company')}")
    return reasons


def _check_all_reachability_perfect(redrob_signals: dict) -> list[str]:
    """recruiter_response_rate, interview_completion_rate, and
    offer_acceptance_rate ALL simultaneously at the theoretical maximum
    (1.0) is the kind of "too good to be true" combination the doc
    describes - a real person's behavior has natural variance."""
    rr = redrob_signals.get("recruiter_response_rate")
    icr = redrob_signals.get("interview_completion_rate")
    oar = redrob_signals.get("offer_acceptance_rate")
    if rr == 1.0 and icr == 1.0 and oar == 1.0:
        return ["recruiter_response_rate, interview_completion_rate, and "
                "offer_acceptance_rate are ALL simultaneously 1.0"]
    return []


# ---------------------------------------------------------------------------
# NEW HARD checks (5 additional impossible-profile patterns)
# ---------------------------------------------------------------------------

# def _check_skill_duration_exceeds_total_experience(
#     profile: dict, skills: list[dict]
# ) -> list[str]:
#     """
#     A single skill's duration_months cannot exceed the candidate's total
#     years_of_experience × 12. You cannot have used Python for 15 years if
#     you've only been working for 5 years total.

#     Note: we tolerate up to 6 months of rounding/internship ambiguity.
#     """
#     yoe = profile.get("years_of_experience")
#     if yoe is None:
#         return []

#     max_possible_months = yoe * 12
#     reasons = []
#     for s in skills:
#         skill_months = s.get("duration_months")
#         if skill_months is not None and skill_months > max_possible_months + 6:
#             reasons.append(
#                 f"skill '{s.get('name')}' claims {skill_months} months of use "
#                 f"but total experience is only {yoe} yrs (~{max_possible_months} mo)"
#             )
#     return reasons


# def _check_overlapping_jobs(career_history: list[dict]) -> list[str]:
#     """
#     Two full-time roles with genuinely overlapping date ranges is logically
#     impossible for a single person. We parse (start_date, end_date) for
#     every job — treating is_current=True / no end_date as 'today' — and
#     flag any pair whose intervals overlap by more than 1 month (1-month
#     tolerance covers the common 'last month at old job, first month at new
#     job' overlap that is perfectly real).
#     """
#     today = date.today()
#     intervals: list[tuple[date, date, str]] = []

#     for j in career_history:
#         start = _parse_date(j.get("start_date"))
#         if start is None:
#             continue
#         end = _parse_date(j.get("end_date")) if not j.get("is_current") else today
#         if end is None:
#             end = today
#         intervals.append((start, end, j.get("company", "unknown")))

#     reasons = []
#     for i in range(len(intervals)):
#         for k in range(i + 1, len(intervals)):
#             s1, e1, c1 = intervals[i]
#             s2, e2, c2 = intervals[k]
#             # Overlap exists when max(s1,s2) < min(e1,e2)
#             overlap_start = max(s1, s2)
#             overlap_end = min(e1, e2)
#             if overlap_start < overlap_end:
#                 overlap_months = _months_between(overlap_start, overlap_end)
#                 if overlap_months > 1:  # tolerance: 1 month is normal transition
#                     reasons.append(
#                         f"jobs at '{c1}' and '{c2}' overlap by ~{overlap_months} months "
#                         f"({overlap_start} to {overlap_end})"
#                     )
#     return reasons


# def _check_salary_range_inverted(redrob_signals: dict) -> list[str]:
#     """
#     expected_salary_range_inr_lpa.min > max is a logical impossibility —
#     a range where the floor exceeds the ceiling cannot exist. Even accounting
#     for data-entry errors, this is an unambiguous flag for a synthetic profile.
#     """
#     salary = redrob_signals.get("expected_salary_range_inr_lpa", {}) or {}
#     min_sal = salary.get("min")
#     max_sal = salary.get("max")
#     if min_sal is not None and max_sal is not None and min_sal > max_sal:
#         return [
#             f"expected_salary_range_inr_lpa has min ({min_sal} LPA) > max ({max_sal} LPA)"
#         ]
#     return []


# def _check_multiple_current_jobs(career_history: list[dict]) -> list[str]:
#     """
#     More than one job with is_current=True is impossible — a person can only
#     hold one 'current' primary role in the context of a standard job profile.
#     (Freelance portfolios aside, the schema models a single active employer.)
#     """
#     current_jobs = [
#         j.get("company", "unknown")
#         for j in career_history
#         if j.get("is_current") is True
#     ]
#     if len(current_jobs) > 1:
#         return [
#             f"multiple jobs marked is_current=True simultaneously: "
#             f"{', '.join(repr(c) for c in current_jobs)}"
#         ]
#     return []


# Earliest year a valid certification in each technology could possibly exist.
# Source: public release / GA dates of each technology.
_TECH_EARLIEST_CERT_YEAR: dict[str, int] = {
    "langchain": 2023,        # Released Oct 2022; first certs realistically 2023
    "chatgpt": 2023,          # Public Nov 2022; certs from 2023 onward
    "openai api": 2021,       # OpenAI API (GPT-3) went public Jun 2020; certs 2021
    "pinecone": 2022,         # GA 2021; certs realistically 2022
    "qdrant": 2022,           # Open-sourced 2021; certs 2022
    "llamaindex": 2023,       # Released as GPT-Index late 2022; rebranded + certs 2023
    "llama index": 2023,
    "llama-index": 2023,
    "autogpt": 2023,          # Released Apr 2023
    "auto-gpt": 2023,
    "stable diffusion": 2023, # Released Aug 2022; formal certs 2023
    "stablediffusion": 2023,
}


def _check_cert_predates_technology(certifications: list[dict]) -> list[str]:
    """
    Flags certifications whose issue year predates the public existence of the
    technology they cover. E.g. a 'LangChain Certified Developer' cert dated
    2020 is impossible — LangChain didn't exist until late 2022.

    Matching is case-insensitive substring: cert name just needs to CONTAIN
    the technology keyword to be checked.
    """
    reasons = []
    for cert in certifications:
        cert_name = (cert.get("name") or "").lower()
        cert_year = cert.get("year") or cert.get("issue_year")
        if cert_year is None:
            continue

        for tech_keyword, earliest_year in _TECH_EARLIEST_CERT_YEAR.items():
            if tech_keyword in cert_name and cert_year < earliest_year:
                reasons.append(
                    f"certification '{cert.get('name')}' dated {cert_year} but "
                    f"'{tech_keyword}' didn't exist until {earliest_year}"
                )
                break  # one reason per cert is enough; avoid duplicate flags
    return reasons


HARD_CHECKS = [
    ("expert_zero_duration",          lambda r: _check_expert_zero_duration(r.get("skills", []))),
    ("all_assessments_perfect",       lambda r: _check_all_assessments_perfect(r.get("redrob_signals", {}))),
    ("education_date_impossible",     lambda r: _check_education_date_impossible(r.get("education", []))),
    ("career_date_math",              lambda r: _check_career_date_math(r.get("career_history", []))),
    ("all_reachability_perfect",      lambda r: _check_all_reachability_perfect(r.get("redrob_signals", {}))),
    # --- 5 new checks below ---
    # ("skill_duration_exceeds_yoe",    lambda r: _check_skill_duration_exceeds_total_experience(
    #                                       r.get("profile", {}), r.get("skills", []))),
    # ("overlapping_jobs",              lambda r: _check_overlapping_jobs(r.get("career_history", []))),
    # ("salary_range_inverted",         lambda r: _check_salary_range_inverted(r.get("redrob_signals", {}))),
    # ("multiple_current_jobs",         lambda r: _check_multiple_current_jobs(r.get("career_history", []))),
    ("cert_predates_technology",      lambda r: _check_cert_predates_technology(
                                          r.get("certifications", []))),
]


# ---------------------------------------------------------------------------
# SOFT checks — suspicious but not impossible; down-weight, don't exclude
# ---------------------------------------------------------------------------

# Max months a skill could possibly have existed as of the pipeline run date.
# Values derived from public GA / first-stable-release dates of each technology.
# Keys are lowercase; matching is case-insensitive substring at runtime.
_TECH_EXIST_LIMITS: dict[str, int] = {
    # --- Original Stack & Variations ---
    "langchain": 44,
    "langgraph": 29,
    "llamaindex": 43,
    "llama-index": 43,
    "qdrant": 63,
    "pinecone": 65,
    "milvus": 80,
    "qlora": 37,
    "q-lora": 37,
    "bentoml": 86,
    "vector search": 111,
    "vector databases": 111,
    "vector db": 111,
    "huggingface transformers": 91,
    "hf transformers": 91,
    "transformers": 91,
    "fine-tuning llms": 92,
    "llm fine-tuning": 92,
    "llm finetuning": 92,
    "fine-tuning": 92,
    "fine tuning": 92,
    "diffusion models": 72,
    "stable diffusion": 72,
    "pgvector": 62,
    "pg_vector": 62,
    "peft": 65,
    "weaviate": 77,
    # --- LoRA & Prompt Engineering ---
    "lora": 60,
    "low-rank adaptation": 60,
    "prompt engineering": 72,
    "prompt design": 72,
    # --- Popular AI/ML Stack ---
    "pytorch": 117,
    "tensorflow": 127,
    "scikit-learn": 228,
    "sklearn": 228,
    "openai api": 72,
    "openai": 72,
    "chromadb": 40,
    "chroma db": 40,
    "chroma": 40,
    "vllm": 36,
    "ollama": 35,
    "rag": 72,
    "retrieval-augmented generation": 72,
    "mlflow": 96,
    "weights & biases": 100,
    "weights and biases": 100,
    "wandb": 100,
    "w&b": 100,
    "gradio": 85,
    "streamlit": 80,
    "fastapi": 90,
}


def _check_skill_duration_exceeds_tech_age(skills: list[dict]) -> list[str]:
    """
    Flags skills whose claimed duration_months exceeds the maximum months the
    technology has existed. E.g. claiming 5 years of LangChain experience is
    impossible — LangChain has only existed ~44 months.

    Matching is case-insensitive substring: a skill name just needs to CONTAIN
    the technology keyword. We use a 3-month tolerance for rounding/early-access.
    Once a keyword matches, we stop checking further keywords for that skill to
    avoid duplicate flags on the same entry.
    """
    reasons = []
    for s in skills:
        skill_name = (s.get("name") or "").lower()
        skill_months = s.get("duration_months")
        if skill_months is None:
            continue

        for tech_key, max_months in _TECH_EXIST_LIMITS.items():
            if tech_key in skill_name and skill_months > max_months + 3:
                reasons.append(
                    f"skill '{s.get('name')}' claims {skill_months} months of experience "
                    f"but '{tech_key}' has only existed ~{max_months} months"
                )
                break  # one flag per skill entry is enough
    return reasons

def _check_availability_contradiction(redrob_signals: dict) -> list[str]:
    if (
        redrob_signals.get("open_to_work_flag") is False
        and (redrob_signals.get("applications_submitted_30d") or 0) > 8
    ):
        return [f"open_to_work=False but submitted "
                f"{redrob_signals.get('applications_submitted_30d')} applications in 30d"]
    return []


def _check_salary_anomaly(profile: dict, redrob_signals: dict) -> list[str]:
    yoe = profile.get("years_of_experience") or 0
    salary = redrob_signals.get("expected_salary_range_inr_lpa", {}) or {}
    max_sal = salary.get("max")
    if yoe >= 8 and max_sal is not None and max_sal < 10:
        return [f"{yoe}yrs experience but expects max {max_sal} LPA "
                f"(unusually low for seniority)"]
    return []


def _check_experience_vs_career_history(profile: dict, career_history: list[dict]) -> list[str]:
    """years_of_experience should be roughly >= the span implied by career
    history (allowing gaps, so we only flag a LARGE mismatch, not a small one)."""
    yoe = profile.get("years_of_experience")
    total_months = sum(j.get("duration_months", 0) or 0 for j in career_history)
    total_years = total_months / 12
    if yoe is not None and total_years > yoe + 2:  # 2yr tolerance for overlap/rounding
        return [f"career_history implies ~{total_years:.1f}yrs but "
                f"years_of_experience claims only {yoe}"]
    return []


SOFT_CHECKS = [
    ("availability_contradiction",    lambda r: _check_availability_contradiction(r.get("redrob_signals", {}))),
    ("salary_anomaly",                lambda r: _check_salary_anomaly(r.get("profile", {}), r.get("redrob_signals", {}))),
    ("experience_mismatch",           lambda r: _check_experience_vs_career_history(
                                          r.get("profile", {}), r.get("career_history", []))),
    ("skill_duration_exceeds_tech_age", lambda r: _check_skill_duration_exceeds_tech_age(r.get("skills", []))),
]


def evaluate_candidate(record: dict[str, Any]) -> dict:
    hard_reasons = []
    for _, check_fn in HARD_CHECKS:
        hard_reasons.extend(check_fn(record))

    soft_reasons = []
    for _, check_fn in SOFT_CHECKS:
        soft_reasons.extend(check_fn(record))

    return {
        "candidate_id": record["candidate_id"],
        "honeypot_flag": len(hard_reasons) > 0,
        "honeypot_reasons": hard_reasons,
        "contradiction_flag": len(soft_reasons) > 0,
        "contradiction_reasons": soft_reasons,
    }


def detect_honeypots(records: list[dict[str, Any]]) -> dict[str, dict]:
    return {r["candidate_id"]: evaluate_candidate(r) for r in records}


if __name__ == "__main__":
    from io_utils import load_candidates

    records, _ = load_candidates(
        "data/raw/sample_candidates.json", "config/candidate_schema.json"
    )
    results = detect_honeypots(records)

    n_hard = sum(1 for r in results.values() if r["honeypot_flag"])
    n_soft = sum(1 for r in results.values() if r["contradiction_flag"])
    print(f"=== Honeypot detection across {len(results)} sample candidates ===\n")
    print(f"HARD honeypot flags (would be excluded): {n_hard}/{len(results)} "
          f"({100*n_hard/len(results):.1f}%)")
    print(f"SOFT contradiction flags (down-weighted only): {n_soft}/{len(results)} "
          f"({100*n_soft/len(results):.1f}%)")
    print()

    if n_hard:
        print("--- HARD flags (detail) ---")
        for cid, r in results.items():
            if r["honeypot_flag"]:
                print(f"{cid}:")
                for reason in r["honeypot_reasons"]:
                    print(f"  - {reason}")

    print("\n--- SOFT flags (detail) ---")
    for cid, r in results.items():
        if r["contradiction_flag"]:
            print(f"{cid}:")
            for reason in r["contradiction_reasons"]:
                print(f"  - {reason}")

    print(f"\nNote: spec disqualification threshold is >10% HARD honeypots in "
          f"the TOP 100 of your final submission, not across the whole pool. "
          f"This sample has 0 designed honeypots by chance (the full pool has "
          f"~80 out of 100,000 = 0.08%, so a 50-candidate random sample "
          f"often won't contain a clean one) - this script's hard-check logic "
          f"is what matters; verify it correctly against true honeypots once "
          f"you have the full candidates.jsonl.gz file.")