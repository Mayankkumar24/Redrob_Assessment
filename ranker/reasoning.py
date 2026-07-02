"""
ranker/reasoning.py
────────────────────
Reads pre-computed reasoning from top_170.json for top-100 candidates.
Reasoning was generated in Phase 1 review step and stored in top_170.json.
"""

from __future__ import annotations


def get_reasoning(candidate_id: str, top_170_map: dict[str, str]) -> str:
    """
    Returns reasoning string for a candidate from top_170_map.
    Fallback if somehow missing: generic string with candidate_id.
    """
    return top_170_map.get(candidate_id) or f"Candidate {candidate_id} selected from top ML/AI production-experience pool."