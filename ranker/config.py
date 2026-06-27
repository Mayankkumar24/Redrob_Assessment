"""
ranker/config.py
────────────────
Single source of truth for all paths, model names, scoring weights,
and disqualification thresholds.  Change values here; nothing else
needs to be touched.
"""

from pathlib import Path

# ── Project root (one level above this file) ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Artifact paths (produced by precompute phase) ────────────────────────────
PATHS = {
    "faiss_index":    ROOT / "data/processed/candidate_index.faiss",
    "candidate_ids":  ROOT / "data/processed/candidate_ids.npy",
    "candidate_meta": ROOT / "data/processed/candidates_clean.parquet",
    "jd_requirements": ROOT / "config/jd_requirements.json",
    "jd_raw":         ROOT / "data/raw/job_description.md",
    "output_csv":     ROOT / "submission.csv",
}

# ── Embedding model (must match Phase 1) ─────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# bge-small uses a query prefix for retrieval tasks
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Retrieval & ranking ───────────────────────────────────────────────────────
TOP_K_RETRIEVE = 1500   # FAISS top-K before reranking / disqualification
TOP_K_FINAL    = 100    # candidates in final CSV

# ── Composite score weights (must sum to 1.0) ────────────────────────────────
WEIGHTS = {
    "semantic":    0.38,   # cosine similarity vs JD embedding
    "skills":      0.30,   # hard-skill coverage score
    "experience":  0.17,   # seniority & years-of-exp fit
    "behavioral":  0.15,   # redrob_signals aggregate
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

# ── Behavioral signal multiplier bounds ──────────────────────────────────────
# Final composite score is multiplied by this before ranking.
# Signals pull the multiplier between FLOOR and CEIL.
SIGNAL_FLOOR = 0.55   # worst possible multiplier (toxic signal portfolio)
SIGNAL_CEIL  = 1.25   # best possible multiplier  (pristine signal portfolio)

# Signals that are **hard disqualifiers** when below their floor
# (i.e., if the signal value ≤ threshold → candidate is flagged honeypot)
SIGNAL_HARD_FLOOR = {
    "profile_consistency_score": 0.30,
    "employment_gap_pattern":    0.20,
    "skill_endorsement_ratio":   0.15,
}

# ── Honeypot detection thresholds ────────────────────────────────────────────
HONEYPOT = {
    # Grace period (months) allowed for overlapping job dates
    "overlap_grace_months": 1,

    # Max skill duration claims capped to (current_year - inception_year).
    # Claiming more years than the tech has existed → honeypot.
    "tech_inception_year": {
        "kubernetes":   2014,
        "docker":       2013,
        "react":        2013,
        "vue":          2014,
        "angular":      2016,
        "pytorch":      2016,
        "tensorflow":   2015,
        "fastapi":      2018,
        "langchain":    2022,
        "llamaindex":   2022,
        "flutter":      2018,
        "rust":         2015,
        "go":           2012,
        "kafka":        2011,
        "spark":        2014,
        "airflow":      2015,
        "dbt":          2016,
        "snowflake":    2012,
        "databricks":   2013,
    },

    # Salary inversion: if current_salary > expected_salary by more than
    # this fraction, flag as anomaly (0.0 = any inversion is suspicious)
    "salary_inversion_tolerance": 0.05,

    # If ALL behavioral signals are ≥ this percentile → suspicious uniformity
    "perfect_signal_threshold": 0.97,

    # Minimum distinct job tenures required (catches resume fabrication)
    "min_job_count_for_senior": 2,
}
