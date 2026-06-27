"""
ranker/embed.py
───────────────
Embed the job description (or any query text) using the same
BAAI/bge-small-en-v1.5 model that was used in Phase 1.

bge-small requires a query prefix for retrieval tasks:
  "Represent this sentence for searching relevant passages: <text>"

We build the JD query string from jd_requirements.json so the
embedding is dense with the right signals rather than raw Markdown noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, BGE_QUERY_PREFIX, PATHS


# ── Module-level model singleton (load once, reuse) ──────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embed] Loading {EMBEDDING_MODEL} …", flush=True)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_jd(jd_req: dict[str, Any]) -> np.ndarray:
    """
    Build a rich query string from parsed JD requirements and embed it.

    Returns a float32 unit-norm vector of shape (dim,).
    Use with a FAISS IndexFlatIP for cosine similarity.
    """
    query_text = _build_jd_query(jd_req)
    print(f"[embed] JD query ({len(query_text)} chars):\n  {query_text[:200]}…", flush=True)

    model = get_model()

    # bge-small: add query prefix for asymmetric retrieval
    prefixed = BGE_QUERY_PREFIX + query_text

    vec = model.encode(
        prefixed,
        normalize_embeddings=True,   # unit norm → dot product = cosine sim
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Batch-embed arbitrary texts (no query prefix).
    Returns float32 array of shape (N, dim).
    Used when Phase-1 stored raw text fields we need to compare at runtime.
    """
    model = get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
        batch_size=64,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_jd_query(jd_req: dict[str, Any]) -> str:
    """
    Flatten structured JD fields into a single retrieval-optimised string.
    Emphasises hard requirements; de-emphasises nice-to-haves.
    """
    parts: list[str] = []

    if title := jd_req.get("title"):
        parts.append(f"Role: {title}.")

    if summary := jd_req.get("summary"):
        parts.append(summary.strip())

    # Hard-required skills (repeated twice for emphasis)
    req_skills = jd_req.get("required_skills", [])
    if req_skills:
        skill_str = ", ".join(req_skills)
        parts.append(f"Required skills: {skill_str}.")
        parts.append(f"Must have: {skill_str}.")   # repetition boosts similarity

    # Nice-to-have (once only)
    nice = jd_req.get("nice_to_have_skills", [])
    if nice:
        parts.append(f"Preferred: {', '.join(nice)}.")

    # Experience level
    if min_exp := jd_req.get("min_experience_years"):
        parts.append(f"Minimum {min_exp} years of experience required.")

    if seniority := jd_req.get("seniority_level"):
        parts.append(f"Seniority: {seniority}.")

    # Domain / industry context
    if domain := jd_req.get("domain"):
        parts.append(f"Domain: {domain}.")

    if responsibilities := jd_req.get("key_responsibilities"):
        parts.append("Responsibilities: " + " ".join(responsibilities))

    return " ".join(parts)
