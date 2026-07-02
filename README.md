# redrob-ranker

**Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**

Two-phase AI pipeline that ranks 100,000 candidates against a Senior AI Engineer job description and produces a top-100 submission CSV.

---

## Table of Contents

- [Approach](#approach)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [⚡ Quick Start — Run the Full Pipeline Here](#-quick-start--run-the-full-pipeline-here)
- [Reproducing the Submission](#Explanation)
- [Phase 1 — Precompute Pipeline](#phase-1--precompute-pipeline-offline)
- [Phase 2 — Ranker](#phase-2--ranker-runtime)
- [Scoring Design](#scoring-design)
- [Honeypot Detection](#honeypot-detection)
- [Compute Constraints](#compute-constraints)
- [Dependencies](#dependencies)

---

## Approach

The core design decision was to **front-load all heavy computation offline** (Phase 1) so the actual ranking step (Phase 2) runs entirely on CPU in under 5 minutes — even against 100,000 candidates.

**Phase 1 (offline, no time limit):**
Ingest all 100K candidates → build rich embedding texts → embed with `BAAI/bge-base-en-v1.5` → build FAISS index → detect honeypots and compute disqualifier flags → store everything in a Parquet file.

**Phase 2 (runtime, ≤5 min CPU):**
Embed the JD → FAISS top-K retrieval → load precomputed flags from Parquet → composite score (semantic + skills + experience + behavioral signals) → rank → write CSV.

No LLM API calls are made during ranking. No GPU is required at runtime. The model is loaded from local HuggingFace cache.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1 — OFFLINE PRECOMPUTE                 │
│                   (run once, network allowed)                    │
│                                                                  │
│  candidates.jsonl.gz                                             │
│         │                                                        │
│         ▼                                                        │
│  [1.1] io_utils.py        Ingest + validate 100K records        │
│         │                                                        │
│         ▼                                                        │
│  [1.1] flatten.py         JSON → flat Polars DataFrame          │
│         │                                                        │
│         ▼                                                        │
│  [1.2] text_builder.py    Build embedding_text per candidate     │
│         │                                                        │
│         ▼                                                        │
│  [1.3] jd_requirements.py Load parsed JD requirements           │
│         │                                                        │
│         ▼                                                        │
│  [1.4] disqualifiers.py   Compute consulting/hopper/CV flags    │
│         │                                                        │
│         ▼                                                        │
│  [1.5] embed_candidates.py  BAAI/bge-base-en-v1.5              │
│         │                   → candidate_embeddings.npy (N×768)  │
│         │                   → production_experience_scores.json  │
│         ▼                                                        │
│  [1.6] build_faiss_index.py  IndexFlatIP over L2-norm vecs     │
│         │                   → candidate_index.faiss             │
│         ▼                                                        │
│  [1.7] honeypot_detector.py  6 hard rules + 4 soft checks      │
│                              → honeypot_flag                     │
│                              → skill_duration_penalty_score      │
│                              → soft_contradiction_penalty_score  │
│                                                                  │
│  OUTPUT: data/processed/candidates_clean.parquet                │
│          data/processed/candidate_index.faiss                   │
│          data/processed/candidate_embeddings.npy                │
│          data/processed/candidate_ids.npy                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (precomputed artifacts)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 2 — RUNTIME RANKING                       │
│              (CPU only, ≤5 min, no network)                     │
│                                                                  │
│  [2.1] artifacts.py      Load FAISS + Parquet + JD config      │
│         │                                                        │
│         ▼                                                        │
│  [2.2] embed.py          Embed JD query → unit-norm vector      │
│         │                (bge-base loaded from local cache)     │
│         ▼                                                        │
│  [2.3] FAISS search      Top-170 cosine similarity retrieval   │
│         │                                                        │
│         ▼                                                        │
│  [2.4] Honeypot filter   Read honeypot_flag from Parquet        │
│         │                Hard disqualified candidates removed    │
│         ▼                                                        │
│  [2.5] scorer.py         base_score = 0.35 × prod_rescaled
                                      + 0.35 × semantic_score
                                      + 0.30 × skills_score 
                           final_score = max(0.0, min(1.0, base * 
                           beh_multiplier
                            * (1.0 - skill_penalty)))
                                                                  │
│         ▼                                                        │
│  [2.6] reasoning.py      2-3 sentence reasoning per candidate   │
│         │                (specific facts)                       │
│         ▼                                                        │
│  OUTPUT: submission.csv  top-100 ranked candidates              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
redrob-ranker/
├── config/
│   ├── candidate_schema.json               JSON Schema for candidate records
│   └── jd_requirements.json                Parsed JD: required skills, min exp, etc.
│
├── data/
│   ├── processed/                          Generated by Phase 1 (not in git)
│   │   ├── candidates_clean.parquet
│   │   ├── candidate_embeddings.npy
│   │   ├── candidate_ids.npy
│   │   ├── candidate_index.faiss
│   │   ├── ingestion_report.json
│   │   ├── production_experience_scores.json
│   │   └── top170_candidates.json          Top 170 candidates (min-max scaled scores + reasoning)
│   │
│   ├── raw/
│   │   ├── candidates.jsonl                Full 100K candidate profiles
│   │   ├── job_description.md
│   │   └── sample_candidates.json          First 50 candidates (schema reference)
│   │
│   └── sandbox_artifacts/                  Copy of processed artifacts for experimentation
│       ├── candidates_clean.parquet
│       ├── candidate_embeddings.npy
│       ├── candidate_ids.npy
│       ├── candidate_index.faiss
│       ├── ingestion_report.json
│       └── production_experience_scores.json
│
├── precompute/                             Phase 1 — offline pipeline
│   ├── run_pipeline.py                     Orchestrator (Steps 1.1–1.7)
│   ├── io_utils.py                         Ingestion + validation
│   ├── flatten.py                          JSON → Polars DataFrame
│   ├── text_builder.py                     Build embedding texts
│   ├── jd_requirements.py                  Load JD config
│   ├── disqualifiers.py                    Consulting/hopper/CV flags
│   ├── embed_candidates.py                 BAAI/bge-base-en-v1.5 embeddings
│   ├── build_faiss_index.py                FAISS IndexFlatIP
│   ├── honeypot_detector.py                Hard + soft violation detection
│   └── review_honeypots.py                 Manual review helper for flagged profiles
│
├── ranker/                                 Phase 2 — runtime ranking
│   ├── config.py                           Paths, weights, thresholds
│   ├── artifacts.py                        Load precomputed artifacts
│   ├── embed.py                            JD embedding
│   ├── behavioral.py                       23 signals → multiplier
│   ├── skills.py                           Skill matching logic
│   ├── scorer.py                           Composite scoring
│   ├── reasoning.py                        Per-candidate reasoning
│   ├── main.py                             Entry point → submission.csv
│   └── smoke_test.py                       Logic tests (no artifacts needed)
│
├── count_reasons.py                        Utility: count/inspect reasoning outputs
├── enrich_top170.py                        Script: enrich top170 with profile metadata
├── flagged_profiles_HARD_only.json         Honeypot profiles flagged by hard rules
├── sandbox.ipynb                           Exploration notebook
├── validate_submission.py                  Validate submission CSV format & constraints
├── submission.csv                          Final ranked top-100 output
├── Mayank_kumar_metadata.yaml              Submission metadata
├── README.md
└── requirements.txt
```

---

## ⚡ Quick Start — Run the Full Pipeline Here

> **Open in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mayankkumar24/Redrob_Assessment/blob/main/sandbox.ipynb)


### Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.10+** (developed and tested on Python 3.12.2)
- **Git LFS** — if not installed, run this command first:
  ```bash
  git lfs install
  ```

---

### Step 1 — Create a new folder

Create a new folder on your machine where you want to set up the project.
For example, name it `Simulate`.

---

### Step 2 — Create a virtual environment

Open the command prompt inside your new folder (`Simulate`) and run the following commands one by one:

**Create the virtual environment:**
```bash
python -m venv .venv
```

**Activate the virtual environment:**
```bash
.venv\Scripts\Activate
```

---

### Step 3 — Clone the repository

> This may take some time depending on your internet speed.

```bash
git clone https://github.com/Mayankkumar24/Redrob_Assessment.git
```

Then navigate into the cloned folder:
```bash
cd Redrob_Assessment
```

---

### Step 4 — Install dependencies

> Do not interrupt execution even if it appears stuck — some packages take time to download.

```bash
pip install -r requirements.txt
```

---

### Step 5 — Run the ranker

> This is the final command. It will complete in under 5 minutes.

```bash
python -m ranker.main --output Mayank_kumar_submission.csv
```

---

### Step 6 — Validate the output CSV

```bash
python validate_submission.py Mayank_kumar_submission.csv
```

---


### Requirements

- Python 3.10+
- 16 GB RAM (Phase 2 runtime)
- Internet access for Phase 1 only (HuggingFace model download)
- GPU optional for Phase 1 (speeds up embedding), not used in Phase 2

### Install dependencies

```bash
git clone https://github.com/Mayankkumar24/Redrob_Assessment.git
cd cd Redrob_Assessment
pip install -r requirements.txt
```

### First-time model download (Phase 1 only)

`BAAI/bge-base-en-v1.5` (~440 MB) is downloaded automatically on first run of Phase 1 and cached at `~/.cache/huggingface`. Phase 2 loads from cache — no network needed at ranking time.

---

## Reproducing the Submission

### Single-command ranking (Phase 2 only)

If you already have the precomputed artifacts in `data/processed/`:

```bash
python -m ranker.main --output submission.csv
```

This is the **timed step** — runs on CPU only, no network, completes in under 5 minutes.

### Full reproduction from scratch

**Step 1 — Precompute (run once, needs network for model download.):**
`Note if you want to run the precomputation step, first keep the candidates.jsonl.gz file in data/raw (here).  it will take a significant time to generate 100000 embeddings on a 8 core cpu + 16 gb ram. i did these step on google colab utilizing the free T4 GPU. (embeddings generated in 40 mins)

```bash
python precompute/run_pipeline.py \
  --input data/raw/candidates.jsonl.gz \
  --output-dir data/processed
```

On Google Colab with candidates on Drive:

```bash
python precompute/run_pipeline.py \
  --input /content/drive/MyDrive/candidates.jsonl \
  --output-dir /content/drive/MyDrive/redrob_output
```

**Step 2 — Rank (timed, CPU only):**

```bash
python -m ranker.main --output submission.csv
```

**Step 3 — Validate:**

```bash
python validate_submission.py submission.csv
```

---

## Phase 1 — Precompute Pipeline (Offline)

### Step 1.1 — Ingestion

`io_utils.py` reads `.jsonl`, `.jsonl.gz`, or `.json` candidate files, validates against `candidate_schema.json`, and returns clean Python dicts. Emits `ingestion_report.json` with counts of skipped/invalid records.

### Step 1.2 — Text construction

`text_builder.py` constructs a single `embedding_text` string per candidate by concatenating their title, skills, experience summary, and education. This is what gets embedded — not raw JSON — so the model sees human-readable text.

### Step 1.3 — JD requirements

`jd_requirements.py` loads `config/jd_requirements.json` — a structured version of the job description with `required_skills`, `nice_to_have_skills`, `min_experience_years`, `seniority_level`, and `domain`. This drives both the embedding query and the skills scoring in Phase 2.

### Step 1.4 — Disqualifier flags

`disqualifiers.py` computes binary flags per candidate:

| Flag | Meaning |
|---|---|
| `is_consulting_only` | Career entirely in consulting/agency, no product company |
| `is_job_hopper` | Average tenure < 12 months across career |
| `cv_speech_only_flag` | Profile language matches CV-writing/public-speaking patterns, not engineering |

These are soft signals used in scoring, not hard disqualifiers.

### Step 1.5 — Embeddings

`embed_candidates.py` encodes all 100K candidates using `BAAI/bge-base-en-v1.5` with L2 normalisation. Also computes a `production_experience_score` per candidate by comparing each embedding against two anchor sentences — one describing production ML system ownership, one describing academic research only. This operationalises the JD's explicit preference for engineers who ship, not just research.

Output: `candidate_embeddings.npy` (shape: N×768, float32).

### Step 1.6 — FAISS Index

`build_faiss_index.py` builds a `faiss.IndexFlatIP` (inner product) over the L2-normalised embeddings. Because vectors are unit-norm, inner product equals cosine similarity. IndexFlatIP was chosen over IVF/HNSW because:
- Exact search (no approximation error)
- 100K vectors at 768 dims fits comfortably in 16 GB RAM (~300 MB)
- Retrieval of top-1500 from 100K takes ~200ms on CPU

### Step 1.7 — Honeypot Detection

See [Honeypot Detection](#honeypot-detection) section below.

---

## Phase 2 — Ranker (Runtime)

Phase 2 runs entirely on CPU with no network access. It reads the precomputed artifacts from Phase 1 and produces the final `submission.csv` in under 5 minutes. The pipeline has 12 steps.

### Step 1 — Load top_170.json

Loads `data/processed/top170_candidates.json` — the 170 candidates pre-selected from Phase 1 by production experience score. Each entry carries a `candidate_id`, a min-max scaled `production_experience_score`, and a pre-computed `reasoning` string (current title, years of experience, AI skill count, recruiter response rate, saved-by-recruiters count).

### Step 2 — Load candidates_clean.parquet

Loads the full 100K candidate metadata from `candidates_clean.parquet` generated in Phase 1. This contains flattened profile fields, skills, behavioral signals, and honeypot flags for every candidate.

### Step 3 — Filter honeypots

Candidates with `honeypot_flag = True` in the parquet are removed from the pool before scoring. Typically ~80 honeypots are identified. If fewer than 100 candidates survive, a warning is emitted.

### Step 4 — Re-scale production scores

The `production_experience_score` values of the surviving pool are min-max re-scaled within that pool to `[0, 1]` and stored as `prod_rescaled`. This ensures the production signal is on the same scale as the other components regardless of how many honeypots were removed.

### Step 5 — Embed the JD

The job description is embedded using `BAAI/bge-base-en-v1.5` (loaded from local HuggingFace cache — no network call) with the BGE query prefix:

```
"Represent this sentence for searching relevant passages: "
```

A compact 512-token JD text (`JD_SHORT_TEXT` in `config.py`) is used to stay within the model's context window. The result is a single 768-dimensional unit-norm vector.

### Step 6 — Compute semantic scores for top-170

Rather than running a full FAISS search over 100K candidates, the pre-computed `candidate_embeddings.npy` is indexed by position using `candidate_ids.npy`. Embeddings for only the 170 pool candidates are loaded and their cosine similarity with the JD vector is computed directly. This keeps Phase 2 fast while still leveraging the same embedding space as Phase 1.

### Step 7 — Build metadata lookup

The parquet is filtered to the pool IDs and converted to a Python dict keyed by `candidate_id` for O(1) access during scoring.

### Step 8 — Score each candidate

Each candidate receives a composite score from three components:

```
base_score  = 0.35 × prod_rescaled
            + 0.35 × semantic_score
            + 0.30 × skills_score

adjusted    = base_score × behavioral_multiplier
final_score = clamp(adjusted × (1 − skill_duration_penalty), 0.0, 1.0)
```

**Production score (`prod_rescaled`)** — pre-computed in Phase 1 by comparing each candidate's embedding against two anchor sentences: one describing production ML system ownership, one describing academic research. Rescaled within the surviving pool.

**Semantic score** — cosine similarity between the candidate's Phase 1 embedding and the JD embedding computed in Step 5. Directly measures holistic fit for this specific JD.

**Skills score (`skills.py`)** — computed from two sources:
- `skill_assessment_scores` from `redrob_signals` (verified scores, normalised to `[0, 1]`)
- Skill name presence in the candidate's profile (gives a 0.50 presence bonus when no assessment score exists)

Scoring formula: `0.80 × must_have_avg + 0.20 × nice_have_avg` across the 4 must-have and 5 nice-to-have JD skill categories defined in `config.py`. Matching uses normalised substring comparison to handle aliases (`python3` → `python`, `chromadb` → `chroma`, etc.).

**Behavioral multiplier (`behavioral.py`)** — a multiplicative factor in `[0.55, 1.30]` computed from 8 signals drawn from `redrob_signals`:

| Signal | Effect |
|---|---|
| `skill_assessment_scores` (avg of JD must-have skills) | +0.06 to +0.12 |
| `github_activity_score` | +0.04 to +0.08, or −0.03 if absent |
| `recruiter_response_rate` | +0.03 to +0.06 |
| `open_to_work_flag` | +0.05 if true, −0.10 if false |
| `last_active_date` (staleness) | −0.05 to −0.20 |
| `interview_completion_rate` | −0.06 to −0.15 |
| `avg_response_time_hours` | −0.04 to −0.10 |
| All three verifications false | −0.05 |

Using a multiplier rather than an additive term ensures behavioral signals can meaningfully suppress a candidate — a high semantic score cannot paper over poor engagement.

**Skill duration penalty** — a magnitude-aware penalty applied if a candidate claims more years with a technology than the technology has existed (e.g., 12 years of Kubernetes, which launched in 2014). Severity is graded by overshoot ratio: mild (≤1.5×), moderate (≤3.0×), severe (>3.0×). Penalty ranges from 0.08 to 0.60 depending on violation count and severity.

### Step 9 — Sort and take top-100

Candidates are sorted descending by `final_score`, with `candidate_id` ascending as a tie-breaker. The top 100 are retained.

### Step 10 — Stretch scores

Final scores are linearly stretched to `[0.05, 0.98]` to improve separation for downstream display:

```
stretched = ((score − min) / (max − min)) × 0.93 + 0.05
```

### Step 11 — Attach reasoning

The pre-computed `reasoning` string for each candidate is read directly from `top170_candidates.json` (populated during the Phase 1 enrichment step). Format:

```
<Current Title> with <X> yrs; <N> AI core skills; response rate <R>; saved by recruiters <S>
```

### Step 12 — Write submission.csv

Outputs `submission.csv` with columns: `candidate_id`, `rank`, `score`, `reasoning`.

---

## Scoring Design

### Why these weights?

| Component | Weight | Rationale |
|---|---|---|
| Production score | 0.35 | Phase 1 anchor-diff directly measures production ML orientation — the JD's primary filter |
| Semantic score | 0.35 | Captures holistic fit for this specific JD in one dense signal; complements the production anchor |
| Skills score | 0.30 | Explicit coverage check for exact JD tools; semantic alone misses precise skill gaps |

### Why multiplicative behavioral multiplier?

An additive behavioral term can be washed out by a strong base score. Making it multiplicative ensures that a candidate with poor behavioral signals (ghosting interviews, very slow response, inactive profile) is genuinely suppressed — not just nudged by a few points. The `[0.55, 1.30]` range means the worst behavioral profile halves the base score, while the best adds a 30% boost.

---

## Honeypot Detection

`honeypot_detector.py` implements two tiers:

### Hard disqualification (6 rules)

Candidates matching any hard rule are excluded from ranking entirely.

| Rule | Description |
|---|---|
| Impossible skill duration | Skill claimed for more years than the technology has existed (e.g., 12 years of Kubernetes, which launched in 2014) |
| Overlapping employment | Two full-time jobs with date ranges overlapping by more than 1 month |
| Salary inversion | `current_salary > expected_salary × 1.05` |
| Cert anachronism | Certification earned before the programme existed, or when candidate was under 14 |
| Perfect signal uniformity | All 23 behavioral signals ≥ 97th percentile simultaneously |
| Hard signal floor | `profile_consistency_score`, `employment_gap_pattern`, or `skill_endorsement_ratio` below catastrophic floors |

### Soft penalties (2 graduated scales)

Candidates are not excluded but their final score is multiplied down.

**Skill duration penalty** — based on violation count across all skills:

| Violations | Penalty |
|---|---|
| 0 | 0.00 |
| 1 | 0.15 |
| 2 | 0.35 |
| 3+ | 0.55 |

**Soft contradiction penalty** — based on how many of 3 checks trigger (availability contradiction, salary anomaly, experience vs career history mismatch):

| Triggered checks | Penalty |
|---|---|
| 0 | 0.00 |
| 1 | 0.10 |
| 2 | 0.20 |
| 3 | 0.40 |

Both penalties are applied multiplicatively and are precomputed in Phase 1, stored in `candidates_clean.parquet`.

---

## Compute Constraints

| Constraint | Limit | Actual |
|---|---|---|
| Runtime (ranking step) | ≤ 5 min | ~90–120 sec on 16GB CPU |
| Memory | ≤ 16 GB | ~2.5 GB peak (FAISS + Parquet + model) |
| Compute | CPU only | ✅ No GPU used in Phase 2 |
| Network | Off | ✅ Model loaded from local cache |
| Disk | ≤ 5 GB | ~1.2 GB (embeddings + index + parquet) |

Phase 1 (precompute) exceeds the 5-minute window — this is expected and permitted. Only the Phase 2 ranking step must satisfy the constraints.

---

## Dependencies

```
polars
numpy
faiss-cpu
sentence-transformers
```

See `requirements.txt` for pinned versions.

---

## Smoke Test

Run all ranker logic tests without needing real artifacts:

```bash
python -m ranker.smoke_test
```

All 6 checks should pass. This validates config weights, behavioral scoring, honeypot column reads, composite scorer, reasoning generator, and CSV format.
