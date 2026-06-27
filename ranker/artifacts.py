"""
ranker/artifacts.py
────────────────────
Load all precomputed Phase-1 artifacts into memory once.
Returns lightweight objects that the ranker reuses across calls.

Expected artifacts on disk
──────────────────────────
data/processed/
  faiss_index.bin        – FAISS IndexFlatIP (L2-normalised embeddings → cosine)
  candidate_ids.json     – list[str], position i → candidate_id
  candidate_meta.parquet – Polars DataFrame with all candidate fields
config/
  jd_requirements.json   – parsed JD: required_skills, nice_to_have, min_exp, etc.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import polars as pl
import numpy as np
from .config import PATHS


@dataclass
class Artifacts:
    faiss_index:    faiss.Index
    candidate_ids:  list[str]           # parallel to FAISS vectors
    meta:           pl.DataFrame        # full candidate table (lazy-friendly)
    jd_req:         dict[str, Any]      # parsed JD requirements


def load_artifacts(verbose: bool = True) -> Artifacts:
    """
    Load all Phase-1 artifacts.  Raises FileNotFoundError with a helpful
    message if any artifact is missing (run precompute first).
    """
    _check_files()

    t0 = time.perf_counter()

    if verbose:
        print("[artifacts] Loading FAISS index …", flush=True)
    index = faiss.read_index(str(PATHS["faiss_index"]))

    if verbose:
        print(f"[artifacts]   {index.ntotal:,} vectors  dim={index.d}", flush=True)

    if verbose:
        print("[artifacts] Loading candidate ID map …", flush=True)
    with open(PATHS["candidate_ids"]) as f:
        # NAYA
        candidate_ids: list[str] = np.load(
        PATHS["candidate_ids"], allow_pickle=True).tolist()

    if index.ntotal != len(candidate_ids):
        raise ValueError(
            f"FAISS index has {index.ntotal} vectors but "
            f"candidate_ids.json has {len(candidate_ids)} entries. "
            "Re-run precompute."
        )

    if verbose:
        print("[artifacts] Loading candidate metadata (Polars) …", flush=True)
    meta = pl.read_parquet(PATHS["candidate_meta"])

    if verbose:
        print(f"[artifacts]   {meta.height:,} rows  {meta.width} columns", flush=True)

    if verbose:
        print("[artifacts] Loading JD requirements …", flush=True)
    with open(PATHS["jd_requirements"]) as f:
        jd_req: dict[str, Any] = json.load(f)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"[artifacts] All artifacts loaded in {elapsed:.2f}s", flush=True)

    return Artifacts(
        faiss_index=index,
        candidate_ids=candidate_ids,
        meta=meta,
        jd_req=jd_req,
    )


def _check_files() -> None:
    missing = []
    for key, path in PATHS.items():
        if key == "output_csv":
            continue  # output, not input
        if not Path(path).exists():
            missing.append(f"  {key}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing precomputed artifacts — run precompute phase first:\n"
            + "\n".join(missing)
        )
