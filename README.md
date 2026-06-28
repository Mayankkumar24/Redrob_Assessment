# redrob-ranker

**Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**

Two-phase AI pipeline that ranks 100,000 candidates against a Senior AI Engineer job description and produces a top-100 submission CSV.

---

## Table of Contents

- [Approach](#approach)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Reproducing the Submission](#reproducing-the-submission)
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
│  [1.7] honeypot_detector.py  6 hard rules + 3 soft checks      │
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
│  [2.3] FAISS search      Top-1500 cosine similarity retrieval   │
│         │                                                        │
│         ▼                                                        │
│  [2.4] Honeypot filter   Read honeypot_flag from Parquet        │
│         │                Hard disqualified candidates removed    │
│         ▼                                                        │
│  [2.5] scorer.py         Composite score per candidate:         │
│         │                  semantic   × 0.38                    │
│         │                + skills     × 0.30                    │
│         │                + experience × 0.17                    │
│         │                + behavioral × 0.15                    │
│         │                × behavioral_multiplier [0.55–1.25]    │
│         │                × (1 - skill_duration_penalty)         │
│         │                × (1 - soft_contradiction_penalty)     │
│         ▼                                                        │
│  [2.6] reasoning.py      1-2 sentence reasoning per candidate   │
│         │                (specific facts, no templates)         │
│         ▼                                                        │
│  OUTPUT: submission.csv  top-100 ranked candidates              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
redrob-ranker/
├── config/
│   ├── candidate_schema.json       JSON Schema for candidate records
│   └── jd_requirements.json        Parsed JD: required skills, min exp, etc.
│
├── data/
│   ├── processed/                  Generated by Phase 1 (not in git)
│   │   ├── candidates_clean.parquet
│   │   ├── candidate_embeddings.npy
│   │   ├── candidate_ids.npy
│   │   ├── candidate_index.faiss
│   │   ├── production_experience_scores.json
│   │   └── ingestion_report.json
│   └── raw/
│       ├── job_description.md
│       └── sample_candidates.json  First 50 candidates (schema reference)
│
├── precompute/                     Phase 1 — offline pipeline
│   ├── run_pipeline.py             Orchestrator (Steps 1.1–1.7)
│   ├── io_utils.py                 Ingestion + validation
│   ├── flatten.py                  JSON → Polars DataFrame
│   ├── text_builder.py             Build embedding texts
│   ├── jd_requirements.py          Load JD config
│   ├── disqualifiers.py            Consulting/hopper/CV flags
│   ├── embed_candidates.py         BAAI/bge-base-en-v1.5 embeddings
│   ├── build_faiss_index.py        FAISS IndexFlatIP
│   └── honeypot_detector.py        Hard + soft violation detection
│
├── ranker/                         Phase 2 — runtime ranking
│   ├── config.py                   Paths, weights, thresholds
│   ├── artifacts.py                Load precomputed artifacts
│   ├── embed.py                    JD embedding
│   ├── behavioral.py               23 signals → multiplier
│   ├── scorer.py                   Composite scoring
│   ├── reasoning.py                Per-candidate reasoning
│   ├── main.py                     Entry point → submission.csv
│   └── smoke_test.py               Logic tests (no artifacts needed)
│
├── README.md
├── requirements.txt
└── submission_metadata.yaml
```

---
|__________________________________________________________________________________________|
## Setup                                                                                   
| -Execute each and every step to get csv file (it is test on my system)
|
|# These are very Curical step for runnging the script into your sandbox.
|
## Prerequisites
|You need this before running any code. You should have a git installed on your system.
|- Python 3.10+
|- Git LFS (git lfs install — one-time setup)
|
# Step 1 — clone into a fresh folder
|git clone https://github.com/Mayankkumar24/Redrob_Assessment.git
|cd Redrob_Assessment
|
# Step 2 — create virtual environment (OPTIONAL BUT RECOMMENDED)
|python -m venv .venv   (USE COMMAND PROMPT)
|.venv\Scripts\activate (USE COMMAND PROMPT)
|
# Step 3 — install dependencies (it will take some time. do not stop execution.)
|pip install -r requirements.txt
|
# Step 4 — MOST IMPORTANT COMMAND TO GET A CSV FILE.
|python -m ranker.main --output submission.csv
|___________________________________________________________________________________________|


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

### JD Embedding

The JD is embedded using the same `bge-base-en-v1.5` model with a query prefix (`"Represent this sentence for searching relevant passages: "`). The query string is constructed from `jd_requirements.json` — required skills are repeated twice for emphasis.

### FAISS Retrieval

Top-1500 candidates retrieved by cosine similarity. This overshoots significantly to ensure the true top-100 survive after honeypot removal and penalty application.

### Composite Scoring

```
raw_score = 0.38 × semantic_score
          + 0.30 × skills_score
          + 0.17 × experience_score
          + 0.15 × behavioral_score

final_score = raw_score
            × behavioral_multiplier          # [0.55, 1.25]
            × (1 - skill_duration_penalty)   # 0 / 0.15 / 0.35 / 0.55
            × (1 - soft_contradiction_penalty) # 0 / 0.10 / 0.20 / 0.40
```

**Semantic score** — cosine similarity from FAISS, shifted from [-1,1] to [0,1].

**Skills score** — 80% from required skill coverage + 20% from nice-to-have coverage. Matching uses normalised substring match to handle `python3` vs `python`, `k8s` vs `kubernetes`, etc.

**Experience score** — trapezoid function: linear ramp below `min_experience_years`, plateau between min and max, small penalty for over-experienced candidates (likely to churn).

**Behavioral score** — 23 `redrob_signals` values bucketed into 5 themes (consistency, engagement, growth, integrity, culture_fit) with weighted aggregation. Bucket weights shift for senior/lead roles to up-weight culture_fit and growth.

### Behavioral Multiplier

Rather than additive scoring alone, behavioral signals apply a multiplicative boost/penalty in [0.55, 1.25]. This means a candidate with identical hard skills but poor behavioral signals is meaningfully penalised, not just nudged.

### Reasoning Generation

Per-candidate reasoning references specific facts — matched skill count, years of experience vs JD requirement, strongest behavioral bucket with its score. Avoids generic phrases. Each reasoning is structurally different because it draws from each candidate's actual profile data.

---

## Scoring Design

### Why these weights?

| Component | Weight | Rationale |
|---|---|---|
| Semantic | 0.38 | Captures holistic fit — skills, domain, tone — in one dense signal |
| Skills | 0.30 | Hard requirements from JD need explicit coverage check; semantic alone misses exact skill gaps |
| Experience | 0.17 | Important but already partially captured by semantic; prevents over-weighting seniority |
| Behavioral | 0.15 | Strong signal but noisier than skills; kept lower to avoid over-trusting engagement proxies |

### Why multiplicative behavioral multiplier?

An additive behavioral term can be washed out by strong semantic similarity. Making it multiplicative ensures that a candidate with toxic behavioral signals (e.g., `profile_consistency_score` near zero) is genuinely suppressed, not just slightly adjusted.

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
