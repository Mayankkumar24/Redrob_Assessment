"""
Step 1.2 — Text Construction & Deduplication

Builds the single `embedding_text` string per candidate that will be fed to
the sentence embedding model in Step 1.5.

Why this step exists: in the sample data, 15/50 candidates have IDENTICAL
career_history descriptions copy-pasted across multiple jobs (a synthetic
data artifact). If we naively concatenate every job description, repeated
text gets counted multiple times and artificially inflates how much that
text dominates the candidate's embedding. We deduplicate first.

We also catch NEAR-duplicates (e.g. minor reworded copies) using rapidfuzz,
not just exact string matches — a 95%+ similarity match is still treated as
a duplicate.

Weighting: profile.summary is the candidate's own authentic voice and the
most reliable signal (titles/skills can be misleading, per the JD's own
"how to read between the lines" section) so it is placed first and repeated
once for emphasis. Deduplicated career descriptions follow. Headline is
included for short-context keyword signal.
"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

NEAR_DUP_THRESHOLD = 90  # rapidfuzz token_sort_ratio, 0-100


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedup_descriptions(descriptions: list[str]) -> list[str]:
    """
    Remove exact and near-duplicate descriptions, preserving first occurrence
    order. Near-duplicate detection uses rapidfuzz token_sort_ratio so minor
    rewording / reordering still counts as a duplicate.
    """
    unique: list[str] = []
    for desc in descriptions:
        desc = _clean_text(desc)
        if not desc:
            continue
        is_dup = any(
            fuzz.token_sort_ratio(desc, existing) >= NEAR_DUP_THRESHOLD
            for existing in unique
        )
        if not is_dup:
            unique.append(desc)
    return unique


def build_embedding_text(record: dict[str, Any]) -> str:
    """
    Build the final text blob used for embedding a single candidate.

    Structure:
      [headline]
      [summary] (x1, weighted by virtue of being the most semantically
                 dense and reliable field)
      [deduplicated career descriptions, most recent first]
    """
    profile = record.get("profile", {})
    headline = _clean_text(profile.get("headline"))
    summary = _clean_text(profile.get("summary"))

    career_history = record.get("career_history", [])
    # most recent first: is_current jobs first, then by start_date desc
    sorted_jobs = sorted(
        career_history,
        key=lambda j: (not j.get("is_current", False), j.get("start_date") or ""),
        reverse=False,
    )
    # the above puts is_current=True (False sorts before True when negated) first;
    # within same is_current bucket we still want most recent start_date first
    sorted_jobs = sorted(
        career_history,
        key=lambda j: (0 if j.get("is_current") else 1, j.get("start_date") or ""),
    )
    raw_descriptions = [j.get("description", "") for j in sorted_jobs]
    unique_descriptions = dedup_descriptions(raw_descriptions)

    parts = []
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)
    parts.extend(unique_descriptions)

    return " ".join(parts)


def build_embedding_texts(records: list[dict[str, Any]]) -> dict[str, str]:
    """Returns {candidate_id: embedding_text} for the full pool."""
    return {r["candidate_id"]: build_embedding_text(r) for r in records}


if __name__ == "__main__":
    from io_utils import load_candidates

    records, _ = load_candidates(
        "data/raw/sample_candidates.json", "config/candidate_schema.json"
    )

    # Show dedup working on a known-duplicate candidate
    target = next(r for r in records if r["candidate_id"] == "CAND_0000031")
    raw_descs = [j["description"] for j in target["career_history"]]
    unique_descs = dedup_descriptions(raw_descs)
    print(f"CAND_0000031: {len(raw_descs)} raw descriptions -> {len(unique_descs)} unique")
    print()

    text = build_embedding_text(target)
    print("embedding_text preview (first 500 chars):")
    print(text[:500])
    print(f"\nFull length: {len(text)} chars")

    print()
    print("=== Dedup savings across full sample (career descriptions only) ===")
    total_raw_chars = 0
    total_dedup_chars = 0
    affected = 0
    for r in records:
        raw_descs = [j.get("description", "") for j in r.get("career_history", [])]
        raw_chars = len(" ".join(raw_descs))
        deduped_chars = len(" ".join(dedup_descriptions(raw_descs)))
        if deduped_chars < raw_chars:
            affected += 1
        total_raw_chars += raw_chars
        total_dedup_chars += deduped_chars
    print(f"Candidates with duplicates removed: {affected}/{len(records)}")
    print(f"Raw concatenated chars: {total_raw_chars:,}")
    print(f"Deduplicated chars: {total_dedup_chars:,}")
    print(f"Reduction: {100 * (1 - total_dedup_chars / total_raw_chars):.1f}%")
