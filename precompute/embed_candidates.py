"""
Step 1.5 — Candidate Embedding Generation

Encodes every candidate's `embedding_text` (from Step 1.2) into a dense
vector using a local sentence-embedding model. This is the most compute-
heavy offline step - done once, network allowed, no time pressure.

ALSO computes `production_experience_score` here (not in disqualifiers.py)
by reusing the SAME loaded model to embed two anchor sentences
("shipped production ML systems" vs "research only, no deployment") and
comparing each candidate's embedding to both via cosine similarity. This
avoids loading the ~130MB model twice.

IMPORTANT - NETWORK NOTE:
This script needs to download BAAI/bge-small-en-v1.5 from huggingface.co
the first time it runs. That domain is not reachable from network-
restricted sandboxes (including the one used to build/test this repo) -
run this on a machine with normal internet access. After the first run,
the model is cached locally (~/.cache/huggingface) and every subsequent
run - including the actual timed ranking run - loads from that local
cache with NO network call. This is exactly why embedding generation is a
Phase 1 (offline) step and not a Phase 2 (runtime) step.

Outputs (to data/processed/):
  candidate_embeddings.npy   - float32 array, shape (N, 384), L2-normalized
  candidate_ids.npy          - array of candidate_id strings, SAME ORDER as
                                the embeddings array - this ordering is the
                                join key back to the rest of the pipeline
  production_experience_scores.json - {candidate_id: score 0-1}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Anchor sentences for the production-vs-research disqualifier signal.
# Chosen to mirror the JD's own language as closely as possible so the
# comparison is measuring exactly the distinction the JD cares about.
PRODUCTION_ANCHOR = (
    "Shipped and operated production machine learning systems serving real "
    "users at scale. Owned embedding drift, index refresh, retrieval quality "
    "regressions, and on-call reliability for live ranking and search "
    "infrastructure deployed to production."
)
RESEARCH_ONLY_ANCHOR = (
    "Published academic research papers in a university or research lab "
    "setting. Work focused on novel model architectures and benchmark "
    "results, with no production deployment, no real users, and no "
    "operational ownership of a live system."
)


def _load_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    print(f"[embed_candidates] Loading model '{model_name}' "
          f"(downloads on first run, then cached locally)...")
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(
    embedding_texts: dict[str, str],
    model=None,
    batch_size: int = 128,
) -> tuple[np.ndarray, list[str]]:
    """
    Encodes all candidate texts. Returns (embeddings, candidate_ids) where
    embeddings[i] corresponds to candidate_ids[i].
    """
    if model is None:
        model = _load_model()

    candidate_ids = list(embedding_texts.keys())
    texts = [embedding_texts[cid] or "" for cid in candidate_ids]

    print(f"[embed_candidates] Encoding {len(texts):,} candidates "
          f"(batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize so dot product = cosine sim
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32), candidate_ids


def compute_production_experience_scores(
    embedding_texts: dict[str, str],
    candidate_embeddings: np.ndarray,
    candidate_ids: list[str],
    model=None,
) -> dict[str, float]:
    """
    For each candidate, score = cosine_sim(candidate, production_anchor) -
    cosine_sim(candidate, research_anchor), rescaled to roughly [0, 1] via
    a sigmoid-like squashing. Higher = more production-shipping language,
    lower = more research-only language. This directly operationalizes the
    JD's "pure research environments without production deployment" and
    "framework enthusiast" / "ships real systems" distinction.
    """
    if model is None:
        model = _load_model()

    anchors = model.encode(
        [PRODUCTION_ANCHOR, RESEARCH_ONLY_ANCHOR],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    prod_anchor, research_anchor = anchors[0], anchors[1]

    # candidate_embeddings already L2-normalized -> dot product = cosine sim
    prod_sims = candidate_embeddings @ prod_anchor
    research_sims = candidate_embeddings @ research_anchor
    raw_diff = prod_sims - research_sims  # roughly in [-1, 1] in practice [-0.3, 0.3]

    # squash to [0, 1] via min-max over this batch for interpretability
    lo, hi = raw_diff.min(), raw_diff.max()
    scaled = (raw_diff - lo) / (hi - lo + 1e-9)

    return {cid: float(score) for cid, score in zip(candidate_ids, scaled)}


def save_artifacts(
    embeddings: np.ndarray,
    candidate_ids: list[str],
    production_scores: dict[str, float],
    output_dir: str = "data/processed",
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "candidate_embeddings.npy", embeddings)
    np.save(out / "candidate_ids.npy", np.array(candidate_ids, dtype=object))
    with open(out / "production_experience_scores.json", "w", encoding="utf-8") as f:
        json.dump(production_scores, f)

    print(f"[embed_candidates] Saved:")
    print(f"  {out / 'candidate_embeddings.npy'}  shape={embeddings.shape}")
    print(f"  {out / 'candidate_ids.npy'}  ({len(candidate_ids)} ids)")
    print(f"  {out / 'production_experience_scores.json'}")


def run(
    candidates_json_path: str = "data/raw/sample_candidates.json",
    schema_path: str = "config/candidate_schema.json",
    output_dir: str = "data/processed",
):
    from io_utils import load_candidates
    from text_builder import build_embedding_texts

    records, _ = load_candidates(candidates_json_path, schema_path)
    embedding_texts = build_embedding_texts(records)

    model = _load_model()
    embeddings, candidate_ids = generate_embeddings(embedding_texts, model=model)
    production_scores = compute_production_experience_scores(
        embedding_texts, embeddings, candidate_ids, model=model
    )
    save_artifacts(embeddings, candidate_ids, production_scores, output_dir)
    return embeddings, candidate_ids, production_scores


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"\n[embed_candidates] FAILED: {e}", file=sys.stderr)
        print(
            "\nThis step requires downloading 'BAAI/bge-small-en-v1.5' from "
            "huggingface.co on first run. If you're seeing a connection "
            "error, you're likely in a network-restricted environment - run "
            "this script on a machine with normal internet access instead. "
            "See the module docstring for details.",
            file=sys.stderr,
        )
        sys.exit(1)
