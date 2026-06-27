"""
ranker/behavioral.py
─────────────────────
Convert the 23 redrob_signals on each candidate into a single
behavioral multiplier in [SIGNAL_FLOOR, SIGNAL_CEIL].

Scoring approach
────────────────
Each signal is normalised to [0, 1].  Signals are grouped into
thematic buckets with bucket-level weights that reflect their
relevance to the job.  The bucket scores are aggregated into a
composite behavioral score, then linearly rescaled to
[SIGNAL_FLOOR, SIGNAL_CEIL].

Multiplier formula:
    behavioral_score  = Σ (bucket_weight × mean(normalised signals in bucket))
    multiplier        = FLOOR + (CEIL - FLOOR) × behavioral_score

Signal groups (based on redrob_signals_doc.md semantics)
──────────────────────────────────────────────────────────
1. CONSISTENCY (weight 0.30)
   profile_consistency_score, employment_gap_pattern,
   skill_endorsement_ratio, job_tenure_stability

2. ENGAGEMENT (weight 0.25)
   platform_activity_score, response_rate_index,
   application_quality_score, interview_completion_rate

3. GROWTH (weight 0.20)
   skill_progression_velocity, learning_signal_score,
   promotion_trajectory, certification_relevance_score

4. INTEGRITY (weight 0.15)
   reference_quality_score, background_check_proxy,
   salary_expectation_realism, notice_period_reliability

5. CULTURE_FIT (weight 0.10)
   culture_fit_proxy, collaborative_signal,
   communication_quality_score, remote_work_adaptability,
   diversity_of_experience, cross_functional_exposure,
   leadership_potential_score
"""

from __future__ import annotations

from typing import Any

from .config import SIGNAL_FLOOR, SIGNAL_CEIL


# ── Signal bucket definitions ─────────────────────────────────────────────────

SIGNAL_BUCKETS: dict[str, dict] = {
    "consistency": {
        "weight": 0.30,
        "signals": [
            "profile_consistency_score",
            "employment_gap_pattern",
            "skill_endorsement_ratio",
            "job_tenure_stability",
        ],
    },
    "engagement": {
        "weight": 0.25,
        "signals": [
            "platform_activity_score",
            "response_rate_index",
            "application_quality_score",
            "interview_completion_rate",
        ],
    },
    "growth": {
        "weight": 0.20,
        "signals": [
            "skill_progression_velocity",
            "learning_signal_score",
            "promotion_trajectory",
            "certification_relevance_score",
        ],
    },
    "integrity": {
        "weight": 0.15,
        "signals": [
            "reference_quality_score",
            "background_check_proxy",
            "salary_expectation_realism",
            "notice_period_reliability",
        ],
    },
    "culture_fit": {
        "weight": 0.10,
        "signals": [
            "culture_fit_proxy",
            "collaborative_signal",
            "communication_quality_score",
            "remote_work_adaptability",
            "diversity_of_experience",
            "cross_functional_exposure",
            "leadership_potential_score",
        ],
    },
}

assert abs(sum(b["weight"] for b in SIGNAL_BUCKETS.values()) - 1.0) < 1e-6


def behavioral_multiplier(
    signals: dict[str, Any],
    jd_req: dict[str, Any] | None = None,
) -> float:
    """
    Compute the behavioral multiplier for a single candidate.

    Parameters
    ----------
    signals : dict
        The candidate's redrob_signals object (23 keys, values in [0, 1]).
    jd_req : dict, optional
        JD requirements — used to adjust bucket weights if the role
        emphasises specific behavioural traits (e.g., leadership-heavy
        role boosts culture_fit bucket).

    Returns
    -------
    float
        Multiplier in [SIGNAL_FLOOR, SIGNAL_CEIL].
    """
    if not signals:
        # No signal data → neutral multiplier (don't penalise, don't reward)
        return (SIGNAL_FLOOR + SIGNAL_CEIL) / 2.0

    weights = _adjusted_weights(jd_req or {})
    composite = 0.0

    for bucket_name, bucket in SIGNAL_BUCKETS.items():
        bucket_vals = []
        for sig_key in bucket["signals"]:
            val = signals.get(sig_key)
            if val is not None and isinstance(val, (int, float)):
                # Clamp to [0, 1]
                bucket_vals.append(max(0.0, min(1.0, float(val))))

        if bucket_vals:
            bucket_score = sum(bucket_vals) / len(bucket_vals)
        else:
            bucket_score = 0.5  # neutral when data missing

        composite += weights[bucket_name] * bucket_score

    multiplier = SIGNAL_FLOOR + (SIGNAL_CEIL - SIGNAL_FLOOR) * composite
    return round(multiplier, 6)


def behavioral_breakdown(signals: dict[str, Any]) -> dict[str, float]:
    """
    Return per-bucket scores for debugging / reasoning generation.
    """
    breakdown: dict[str, float] = {}
    for bucket_name, bucket in SIGNAL_BUCKETS.items():
        vals = []
        for sig_key in bucket["signals"]:
            val = signals.get(sig_key)
            if val is not None and isinstance(val, (int, float)):
                vals.append(max(0.0, min(1.0, float(val))))
        breakdown[bucket_name] = round(sum(vals) / len(vals), 3) if vals else 0.5
    return breakdown


# ── Weight adjustment for JD context ─────────────────────────────────────────

def _adjusted_weights(jd_req: dict[str, Any]) -> dict[str, float]:
    """
    Optionally boost specific bucket weights based on JD signals.
    Renormalises to sum = 1.0 after adjustment.
    """
    weights = {k: v["weight"] for k, v in SIGNAL_BUCKETS.items()}
    seniority = (jd_req.get("seniority_level") or "").lower()

    # Leadership / senior roles → boost culture_fit & growth
    if any(kw in seniority for kw in ("lead", "senior", "principal", "staff", "head")):
        weights["culture_fit"] = min(0.20, weights["culture_fit"] * 1.5)
        weights["growth"]      = min(0.28, weights["growth"] * 1.3)

    # Re-normalise
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}
