"""
ranker/behavioral.py
─────────────────────
Computes behavioral_multiplier from the REAL 23 redrob_signals.

Only 7+1 signals are used (rest are vanity metrics or logistics):

REWARD signals:
  skill_assessment_scores   → JD must-have skill avg (assessed score)
  github_activity_score     → engineering activity proof
  recruiter_response_rate   → hireability predictor
  open_to_work_flag         → availability gate

PENALTY signals:
  last_active_date          → staleness (months since last login)
  interview_completion_rate → ghosting pattern
  avg_response_time_hours   → poor engagement

TRUST signal (special):
  verified_email + verified_phone + linkedin_connected
  → all three False → small penalty (suspicious legitimacy)

multiplier = clamp(1.0 + sum(all adjustments), FLOOR, CEIL)
           = clamp in [0.55, 1.30]
"""

from __future__ import annotations
import datetime
from typing import Any

from .config import BEHAVIORAL, BEHAVIORAL_FLOOR, BEHAVIORAL_CEIL, MUST_HAVE_SKILLS


def compute_behavioral_multiplier(row: dict[str, Any]) -> float:
    """
    Returns behavioral multiplier in [BEHAVIORAL_FLOOR, BEHAVIORAL_CEIL].
    """
    signals = _get_signals(row)
    if not signals:
        return 1.0   # neutral — no data

    total_adjustment = 0.0
    total_adjustment += _skill_assessment_reward(signals)
    total_adjustment += _github_reward(signals)
    total_adjustment += _response_rate_reward(signals)
    total_adjustment += _open_to_work_adjustment(signals)
    total_adjustment += _staleness_penalty(signals)
    total_adjustment += _interview_completion_penalty(signals)
    total_adjustment += _response_time_penalty(signals)
    total_adjustment += _trust_penalty(signals)

    multiplier = 1.0 + total_adjustment
    return round(max(BEHAVIORAL_FLOOR, min(BEHAVIORAL_CEIL, multiplier)), 6)


def behavioral_detail(row: dict[str, Any]) -> dict[str, float]:
    """
    Returns per-signal adjustments for reasoning generation.
    """
    signals = _get_signals(row)
    return {
        "skill_assessment": _skill_assessment_reward(signals),
        "github":           _github_reward(signals),
        "response_rate":    _response_rate_reward(signals),
        "open_to_work":     _open_to_work_adjustment(signals),
        "staleness":        _staleness_penalty(signals),
        "interview":        _interview_completion_penalty(signals),
        "response_time":    _response_time_penalty(signals),
        "trust":            _trust_penalty(signals),
    }


# ── Individual signal functions ───────────────────────────────────────────────

def _skill_assessment_reward(signals: dict) -> float:
    """
    Average assessment score across JD must-have skill aliases.
    Only uses skills that appear in skill_assessment_scores.
    """
    raw: dict = signals.get("skill_assessment_scores") or {}
    if not isinstance(raw, dict) or not raw:
        return 0.0

    import re
    def norm(t): return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

    # Build flat alias list from must-have categories
    all_aliases: list[str] = []
    for aliases in MUST_HAVE_SKILLS.values():
        all_aliases.extend(aliases)

    matched_scores: list[float] = []
    for skill_name, score in raw.items():
        sn = norm(skill_name)
        for alias in all_aliases:
            if norm(alias) in sn or sn in norm(alias):
                try:
                    matched_scores.append(float(score))
                except (ValueError, TypeError):
                    pass
                break

    if not matched_scores:
        return 0.0

    avg = sum(matched_scores) / len(matched_scores)
    cfg = BEHAVIORAL
    if avg >= cfg["skill_assessment_threshold_high"]:
        return cfg["skill_assessment_reward_high"]
    if avg >= cfg["skill_assessment_threshold_mid"]:
        return cfg["skill_assessment_reward_mid"]
    return 0.0


def _github_reward(signals: dict) -> float:
    score = signals.get("github_activity_score")
    if score is None:
        return 0.0
    try:
        score = float(score)
    except (ValueError, TypeError):
        return 0.0
    cfg = BEHAVIORAL
    if score == -1:
        return cfg["github_penalty_none"]
    if score >= cfg["github_threshold_high"]:
        return cfg["github_reward_high"]
    if score >= cfg["github_threshold_mid"]:
        return cfg["github_reward_mid"]
    return 0.0


def _response_rate_reward(signals: dict) -> float:
    rate = signals.get("recruiter_response_rate")
    if rate is None:
        return 0.0
    try:
        rate = float(rate)
    except (ValueError, TypeError):
        return 0.0
    cfg = BEHAVIORAL
    if rate >= cfg["response_rate_threshold_high"]:
        return cfg["response_rate_reward_high"]
    if rate >= cfg["response_rate_threshold_mid"]:
        return cfg["response_rate_reward_mid"]
    return 0.0


def _open_to_work_adjustment(signals: dict) -> float:
    flag = signals.get("open_to_work_flag")
    if flag is None:
        return 0.0
    cfg = BEHAVIORAL
    if flag is True or str(flag).lower() in ("true", "1", "yes"):
        return cfg["open_to_work_reward"]
    return cfg["open_to_work_penalty"]


def _staleness_penalty(signals: dict) -> float:
    """Days since last_active_date."""
    date_str = signals.get("last_active_date")
    if not date_str:
        return 0.0
    try:
        last = datetime.date.fromisoformat(str(date_str)[:10])
        days_ago = (datetime.date.today() - last).days
    except (ValueError, TypeError):
        return 0.0
    cfg = BEHAVIORAL
    if days_ago > cfg["staleness_threshold_severe"]:
        return cfg["staleness_penalty_severe"]
    if days_ago > cfg["staleness_threshold_high"]:
        return cfg["staleness_penalty_high"]
    if days_ago > cfg["staleness_threshold_mid"]:
        return cfg["staleness_penalty_mid"]
    return 0.0


def _interview_completion_penalty(signals: dict) -> float:
    rate = signals.get("interview_completion_rate")
    if rate is None:
        return 0.0
    try:
        rate = float(rate)
    except (ValueError, TypeError):
        return 0.0
    cfg = BEHAVIORAL
    if rate < cfg["interview_threshold_high"]:
        return cfg["interview_penalty_high"]
    if rate < cfg["interview_threshold_mid"]:
        return cfg["interview_penalty_mid"]
    return 0.0


def _response_time_penalty(signals: dict) -> float:
    hours = signals.get("avg_response_time_hours")
    if hours is None:
        return 0.0
    try:
        hours = float(hours)
    except (ValueError, TypeError):
        return 0.0
    cfg = BEHAVIORAL
    if hours > cfg["response_time_threshold_high"]:
        return cfg["response_time_penalty_high"]
    if hours > cfg["response_time_threshold_mid"]:
        return cfg["response_time_penalty_mid"]
    return 0.0


def _trust_penalty(signals: dict) -> float:
    """Small penalty if ALL three verification signals are False."""
    v_email = signals.get("verified_email")
    v_phone = signals.get("verified_phone")
    v_li    = signals.get("linkedin_connected")

    def is_false(val):
        if val is None:
            return False
        if val is False:
            return True
        return str(val).lower() in ("false", "0", "no")

    if is_false(v_email) and is_false(v_phone) and is_false(v_li):
        return BEHAVIORAL["trust_all_false_penalty"]
    return 0.0


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_signals(row: dict) -> dict:
    signals = row.get("redrob_signals")
    if isinstance(signals, dict):
        return signals
    if isinstance(signals, str):
        import json
        try:
            return json.loads(signals)
        except Exception:
            pass
    return {}
