"""
ranker/config.py
────────────────
Single source of truth for all paths, weights, thresholds,
JD text, and skill mappings.
"""

from pathlib import Path
import datetime

ROOT         = Path(__file__).resolve().parent.parent
CURRENT_YEAR = datetime.date.today().year

# ── Paths ─────────────────────────────────────────────────────────────────────
PATHS = {
    "top_170":          ROOT / "data/processed/top170_candidates.json",
    "candidate_meta":   ROOT / "data/processed/candidates_clean.parquet",
    "candidate_emb":    ROOT / "data/processed/candidate_embeddings.npy",
    "candidate_ids":    ROOT / "data/processed/candidate_ids.npy",
    "output_csv":       ROOT / "submission.csv",
}

# ── Embedding model (must match Phase 1) ─────────────────────────────────────
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Short JD text (fits within 512-token bge-base context window) ─────────────
JD_SHORT_TEXT = (
    "Redrob AI is recruiting a Senior AI Engineer for its founding team to own "
    "candidate-JD matching, ranking, and retrieval systems. The position requires "
    "migrating from BM25 and rule-based scoring to an advanced system using "
    "embeddings, hybrid retrieval, and LLM-based re-ranking. Mandatory requirements "
    "include strong Python and production experience with embeddings-based retrieval "
    "systems (sentence-transformers, OpenAI, BGE, E5) handling index refresh and "
    "embedding drift, plus operational familiarity with vector databases (Pinecone, "
    "Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). Candidate must "
    "design rigorous evaluation frameworks using NDCG, MRR, MAP, and online A/B "
    "testing. Preferred skills cover LLM fine-tuning (LoRA, QLoRA, PEFT) and "
    "learning-to-rank systems. The role explicitly excludes pure researchers without "
    "production deployments, non-coding architects, LangChain-only developers, and "
    "candidates with exclusive consulting firm experience. Target profile is 5-9 "
    "years total experience, with 4-5 years in applied ML/AI roles at product "
    "companies shipping end-to-end ranking, search, or recommendation systems to "
    "real users at scale."
)

# ── Scoring weights ───────────────────────────────────────────────────────────
WEIGHT_PRODUCTION = 0.35
WEIGHT_SEMANTIC   = 0.35
WEIGHT_SKILLS     = 0.30

WEIGHT_MUST_HAVE  = 0.80
WEIGHT_NICE_HAVE  = 0.20

TOP_K_FINAL = 100

# ── JD Skill mappings ─────────────────────────────────────────────────────────
MUST_HAVE_SKILLS = {
    "embeddings": [
        "embeddings", "embedding", "sentence-transformers",
        "sentence_transformers", "openai embeddings", "bge", "e5",
        "text embeddings", "dense retrieval", "bi-encoder",
    ],
    "vector_database": [
        "vector database", "vector db", "vector store",
        "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch", "faiss", "annoy",
        "pgvector", "chroma", "chromadb", "vespa",
        "hybrid search", "approximate nearest neighbor", "ann",
    ],
    "python": [
        "python", "python3", "python 3",
    ],
    "ranking_evaluation": [
        "ndcg", "mrr", "map", "mean average precision",
        "mean reciprocal rank", "ranking evaluation",
        "offline evaluation", "a/b testing", "ab testing",
        "retrieval evaluation", "evaluation framework",
        "learning to rank", "ltr",
    ],
}

NICE_TO_HAVE_SKILLS = {
    "llm_finetuning": [
        "lora", "qlora", "peft", "fine-tuning", "finetuning",
        "fine tuning", "llm fine-tuning", "instruction tuning", "rlhf",
    ],
    "learning_to_rank": [
        "learning to rank", "ltr", "xgboost ranking",
        "lambdamart", "ranknet", "neural ltr",
    ],
    "hr_tech": [
        "hr tech", "hrtech", "recruiting", "recruitment tech",
        "ats", "applicant tracking", "talent acquisition", "marketplace",
    ],
    "distributed_systems": [
        "distributed systems", "distributed computing",
        "large scale inference", "inference optimization",
        "model serving", "mlops", "triton", "ray",
    ],
    "open_source": [
        "open source", "open-source", "opensource",
        "github contributions", "open source contributions",
    ],
}

# ── Tech inception years (skill duration penalty) ─────────────────────────────
TECH_INCEPTION_YEAR = {
    "kubernetes":    2014,
    "docker":        2013,
    "react":         2013,
    "vue":           2014,
    "angular":       2016,
    "pytorch":       2016,
    "tensorflow":    2015,
    "fastapi":       2018,
    "langchain":     2022,
    "llamaindex":    2022,
    "llama index":   2022,
    "flutter":       2018,
    "rust":          2015,
    "kafka":         2011,
    "spark":         2014,
    "airflow":       2015,
    "dbt":           2016,
    "snowflake":     2012,
    "databricks":    2013,
    "transformers":  2017,
    "hugging face":  2016,
    "huggingface":   2016,
    "openai api":    2020,
    "chatgpt":       2022,
    "gpt-4":         2023,
    "qdrant":        2021,
    "weaviate":      2019,
    "pinecone":      2019,
    "milvus":        2019,
    "chromadb":      2022,
    "chroma":        2022,
    "lora":          2021,
    "qlora":         2023,
    "peft":          2022,
}

# ── Skill duration penalty table ──────────────────────────────────────────────
# (min(violation_count, 3), worst_severity) → penalty fraction
SKILL_DURATION_PENALTY = {
    (1, "mild"):     0.08,
    (1, "moderate"): 0.15,
    (1, "severe"):   0.25,
    (2, "mild"):     0.20,
    (2, "moderate"): 0.30,
    (2, "severe"):   0.40,
    (3, "mild"):     0.35,
    (3, "moderate"): 0.45,
    (3, "severe"):   0.60,
}

# ── Behavioral multiplier bounds ──────────────────────────────────────────────
BEHAVIORAL_FLOOR = 0.55
BEHAVIORAL_CEIL  = 1.30

# ── Behavioral signal thresholds ─────────────────────────────────────────────
BEHAVIORAL = {
    "skill_assessment_reward_high":    0.12,
    "skill_assessment_reward_mid":     0.06,
    "skill_assessment_threshold_high": 80,
    "skill_assessment_threshold_mid":  60,

    "github_reward_high":    0.08,
    "github_reward_mid":     0.04,
    "github_penalty_none":  -0.03,
    "github_threshold_high": 70,
    "github_threshold_mid":  40,

    "response_rate_reward_high":    0.06,
    "response_rate_reward_mid":     0.03,
    "response_rate_threshold_high": 0.7,
    "response_rate_threshold_mid":  0.4,

    "open_to_work_reward":  0.05,
    "open_to_work_penalty": -0.10,

    "staleness_penalty_mid":     -0.05,
    "staleness_penalty_high":    -0.12,
    "staleness_penalty_severe":  -0.20,
    "staleness_threshold_mid":    30,
    "staleness_threshold_high":   90,
    "staleness_threshold_severe": 180,

    "interview_penalty_mid":  -0.06,
    "interview_penalty_high": -0.15,
    "interview_threshold_mid":  0.7,
    "interview_threshold_high": 0.4,

    "response_time_penalty_mid":  -0.04,
    "response_time_penalty_high": -0.10,
    "response_time_threshold_mid":   24,
    "response_time_threshold_high":  72,

    "trust_all_false_penalty": -0.05,
}