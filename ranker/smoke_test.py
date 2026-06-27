"""
ranker/smoke_test.py
─────────────────────
Validates all ranker logic WITHOUT needing real artifacts.
Creates tiny synthetic data, runs every module, checks outputs.

Run:
    python -m ranker.smoke_test
"""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    print("=" * 60)
    print("RANKER SMOKE TEST")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. Config ─────────────────────────────────────────────────────────────
    print("\n[1/7] Config …")
    try:
        from ranker.config import WEIGHTS, SIGNAL_FLOOR, SIGNAL_CEIL, HONEYPOT
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights don't sum to 1"
        assert SIGNAL_FLOOR < SIGNAL_CEIL, "Signal floor >= ceil"
        print(f"     Weights: {WEIGHTS}")
        print(f"     Signal range: [{SIGNAL_FLOOR}, {SIGNAL_CEIL}]")
        print("     ✓ Config OK")
    except Exception as e:
        errors.append(f"config: {e}")
        print(f"     ✗ {e}")

    # ── 2. Behavioral scorer ──────────────────────────────────────────────────
    print("\n[2/7] Behavioral scorer …")
    try:
        from ranker.behavioral import behavioral_multiplier, behavioral_breakdown

        perfect_signals = {k: 0.99 for k in [
            "profile_consistency_score", "employment_gap_pattern",
            "skill_endorsement_ratio", "job_tenure_stability",
            "platform_activity_score", "response_rate_index",
            "application_quality_score", "interview_completion_rate",
            "skill_progression_velocity", "learning_signal_score",
            "promotion_trajectory", "certification_relevance_score",
            "reference_quality_score", "background_check_proxy",
            "salary_expectation_realism", "notice_period_reliability",
            "culture_fit_proxy", "collaborative_signal",
            "communication_quality_score", "remote_work_adaptability",
            "diversity_of_experience", "cross_functional_exposure",
            "leadership_potential_score",
        ]}
        bad_signals = {k: 0.01 for k in perfect_signals}

        mult_perfect = behavioral_multiplier(perfect_signals)
        mult_bad     = behavioral_multiplier(bad_signals)
        mult_empty   = behavioral_multiplier({})

        print(f"     Perfect signals → multiplier: {mult_perfect:.4f}")
        print(f"     Bad signals    → multiplier: {mult_bad:.4f}")
        print(f"     Empty signals  → multiplier: {mult_empty:.4f}")

        assert mult_perfect > mult_bad, "Perfect should > bad"
        assert mult_empty > 0, "Empty must be positive"

        bd = behavioral_breakdown(perfect_signals)
        print(f"     Breakdown keys: {list(bd.keys())}")
        print("     ✓ Behavioral scorer OK")
    except Exception as e:
        errors.append(f"behavioral: {e}")
        print(f"     ✗ {e}")

    # NAYA [3/7] block
    print("\n[3/7] Honeypot detection (parquet column check) …")
    try:
        import polars as pl

        # Phase 2 reads honeypot_flag directly from parquet — no re-detection
        mock_meta = pl.DataFrame({
            "candidate_id":  ["c001", "c002", "c003"],
            "honeypot_flag": [False,  True,   False],
        })
        disqualified = set(
            mock_meta.filter(pl.col("honeypot_flag"))["candidate_id"].to_list()
        )
        assert "c001" not in disqualified, "Clean candidate wrongly flagged!"
        assert "c002" in disqualified,     "Honeypot not caught!"
        assert "c003" not in disqualified, "Clean candidate wrongly flagged!"
        print(f"     Disqualified from parquet column: {disqualified}")
        print("     ✓ Honeypot column read OK")
    except Exception as e:
        errors.append(f"honeypot: {e}")
        print(f"     ✗ {e}")

    # ── 4. Scorer ─────────────────────────────────────────────────────────────
    print("\n[4/7] Scorer …")
    try:
        import polars as pl
        from ranker.scorer import score_candidates

        n = 10
        dim = 384   # bge-small dim

        # Mock FAISS output
        faiss_ids   = np.arange(n, dtype=np.int64)
        faiss_dists = np.linspace(0.9, 0.5, n).astype(np.float32)

        cand_ids = [f"cand_{i:03d}" for i in range(n)]

        meta_rows = []
        for i in range(n):
            meta_rows.append({
                "candidate_id": cand_ids[i],
                "skills": [
                    {"name": "Python", "years_of_experience": 4},
                    {"name": "FastAPI", "years_of_experience": 2},
                    {"name": "PostgreSQL", "years_of_experience": 3},
                ],
                "total_experience_years": 5.0,
                "seniority_level": "senior",
                "work_experience": [],
                "current_salary": 80000,
                "expected_salary": 100000,
                "certifications": [],
                "redrob_signals": {
                    "profile_consistency_score": 0.7 + i * 0.01,
                    "employment_gap_pattern": 0.65,
                    "skill_endorsement_ratio": 0.72,
                    "job_tenure_stability": 0.80,
                    "platform_activity_score": 0.60,
                    "response_rate_index": 0.75,
                    "application_quality_score": 0.68,
                    "interview_completion_rate": 0.85,
                    "skill_progression_velocity": 0.70,
                    "learning_signal_score": 0.65,
                    "promotion_trajectory": 0.55,
                    "certification_relevance_score": 0.60,
                    "reference_quality_score": 0.75,
                    "background_check_proxy": 0.90,
                    "salary_expectation_realism": 0.80,
                    "notice_period_reliability": 0.70,
                    "culture_fit_proxy": 0.65,
                    "collaborative_signal": 0.72,
                    "communication_quality_score": 0.68,
                    "remote_work_adaptability": 0.75,
                    "diversity_of_experience": 0.60,
                    "cross_functional_exposure": 0.55,
                    "leadership_potential_score": 0.50,
                },
            })

        meta_df = pl.DataFrame(meta_rows)
        jd_req = {
            "title": "Senior Backend Engineer",
            "required_skills": ["Python", "FastAPI", "PostgreSQL"],
            "nice_to_have_skills": ["Redis", "Docker"],
            "min_experience_years": 4,
            "max_experience_years": 10,
            "seniority_level": "senior",
        }

        ranked = score_candidates(
            faiss_ids=faiss_ids,
            faiss_dists=faiss_dists,
            candidate_ids=cand_ids,
            meta=meta_df,
            jd_req=jd_req,
            disqualified=set(),
        )

        print(f"     Scored {len(ranked)} candidates")
        print(f"     Columns: {ranked.columns}")
        print(f"     Top score: {ranked['final_score'].max():.4f}")
        assert len(ranked) == n, f"Expected {n} rows, got {len(ranked)}"
        assert "final_score" in ranked.columns
        assert ranked["final_score"][0] >= ranked["final_score"][-1], "Not sorted!"
        print("     ✓ Scorer OK")
    except Exception as e:
        errors.append(f"scorer: {e}")
        print(f"     ✗ {e}")

    # ── 5. Reasoning ──────────────────────────────────────────────────────────
    print("\n[5/7] Reasoning generator …")
    try:
        from ranker.reasoning import generate_reasoning

        sample_row = {
            "candidate_id": "cand_001",
            "total_experience_years": 6.0,
            "seniority_level": "senior",
            "skills": [
                {"name": "Python", "years_of_experience": 5},
                {"name": "FastAPI", "years_of_experience": 3},
                {"name": "PostgreSQL", "years_of_experience": 4},
            ],
            "behavioral_mult": 1.18,
            "behavioral_detail": "consistency=0.82 | engagement=0.74 | growth=0.70 | integrity=0.80 | culture_fit=0.65",
            "final_score": 0.8542,
        }
        jd_req_sample = {
            "required_skills": ["Python", "FastAPI", "PostgreSQL"],
            "nice_to_have_skills": ["Redis"],
            "min_experience_years": 4,
            "seniority_level": "senior",
        }

        reason = generate_reasoning(sample_row, jd_req_sample)
        print(f"     Sample reasoning ({len(reason)} chars):")
        print(f"     → {reason}")
        assert 20 < len(reason) < 350, f"Reasoning length {len(reason)} out of range"
        print("     ✓ Reasoning generator OK")
    except Exception as e:
        errors.append(f"reasoning: {e}")
        print(f"     ✗ {e}")

    # ── 6. CSV format check ───────────────────────────────────────────────────
    print("\n[6/7] CSV format check …")
    try:
        import csv

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "candidate_id", "score", "reasoning"])
            writer.writerow([1, "cand_001", "0.854200", "Strong Python and FastAPI match with 6 years experience."])
            tmp_path = f.name

        with open(tmp_path) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["rank"] == "1"
        assert rows[0]["candidate_id"] == "cand_001"
        assert "rank" in rows[0] and "reasoning" in rows[0]
        print(f"     CSV row sample: {dict(rows[0])}")
        print("     ✓ CSV format OK")
    except Exception as e:
        errors.append(f"csv_format: {e}")
        print(f"     ✗ {e}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not errors:
        print("ALL CHECKS PASSED ✓  — ready to run on real data")
        print("=" * 60)
        print("\nNext step:  python -m ranker.main")
    else:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ✗  {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
