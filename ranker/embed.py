"""
ranker/embed.py
───────────────
Two responsibilities in Phase 2:

1. embed_jd()
   Embeds the short JD text using BAAI/bge-base-en-v1.5 with BGE
   query prefix. Returns unit-norm float32 vector (768-dim).

2. load_candidate_semantic_scores(top_170_ids)
   Extracts embeddings ONLY for the top-170 candidate IDs from the
   precomputed candidate_embeddings.npy (100K × 768) by looking up
   their row indices in candidate_ids.npy.
   Returns dict {candidate_id: cosine_similarity_with_JD}.

Why extract subset:
   Loading full 100K × 768 matrix (~300MB) just to use 170 rows
   wastes RAM. We load the full array once, index into it, then
   release. Cosine sim = dot product since both are L2-normalised.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, BGE_QUERY_PREFIX, JD_SHORT_TEXT, PATHS


# ── Module-level model singleton ──────────────────────────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embed] Loading {EMBEDDING_MODEL} from local cache …", flush=True)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ── 1. JD embedding ───────────────────────────────────────────────────────────

def embed_jd() -> np.ndarray:
    """
    Embeds JD_SHORT_TEXT with BGE query prefix.
    Returns float32 unit-norm vector shape (768,).
    """
    model  = get_model()
    text   = BGE_QUERY_PREFIX + JD_SHORT_TEXT
    vec    = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float32)


# ── 2. Candidate semantic scores ──────────────────────────────────────────────

def load_candidate_semantic_scores(
    top_170_ids: list[str],
    jd_vec: np.ndarray,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Loads candidate_embeddings.npy and candidate_ids.npy,
    extracts rows for top_170_ids, computes cosine similarity
    with jd_vec (dot product, since both L2-normalised).

    Returns {candidate_id: cosine_similarity} in [0, 1].
    Cosine from dot product is in [-1, 1]; we shift to [0, 1].
    """
    emb_path = PATHS["candidate_emb"]
    ids_path = PATHS["candidate_ids"]

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Candidate IDs not found: {ids_path}")

    if verbose:
        print("[embed] Loading candidate_ids.npy …", flush=True)
    all_ids: np.ndarray = np.load(ids_path, allow_pickle=True)

    # Build id → row-index lookup
    id_to_idx: dict[str, int] = {
        str(cid): idx for idx, cid in enumerate(all_ids)
    }

    # Find indices for our top-170
    indices: list[int] = []
    found_ids: list[str] = []
    missing: list[str] = []

    for cid in top_170_ids:
        idx = id_to_idx.get(str(cid))
        if idx is not None:
            indices.append(idx)
            found_ids.append(cid)
        else:
            missing.append(cid)

    if missing:
        print(f"[embed] WARNING: {len(missing)} IDs not found in candidate_ids.npy", flush=True)

    if verbose:
        print(f"[embed] Loading embeddings for {len(indices)} candidates …", flush=True)

    # Load full matrix and extract subset
    # np.load with mmap_mode='r' avoids loading full 300MB into RAM at once
    all_emb = np.load(emb_path, mmap_mode="r")
    subset  = all_emb[indices].astype(np.float32)   # shape (len(indices), 768)

    # Cosine similarity = dot product (both L2-normalised)
    raw_sims: np.ndarray = subset @ jd_vec           # shape (len(indices),)

    # Shift [-1, 1] → [0, 1]
    sims_01 = (raw_sims + 1.0) / 2.0

    scores = {
        cid: float(sim)
        for cid, sim in zip(found_ids, sims_01)
    }

    if verbose:
        print(
            f"[embed] Semantic scores — "
            f"min={min(scores.values()):.4f}  "
            f"max={max(scores.values()):.4f}",
            flush=True,
        )

    return scores