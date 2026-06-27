"""
ranker/main.py
───────────────
Phase 2 entry point.  Orchestrates the full ranking pipeline and
writes the submission CSV.

Usage
─────
    python -m ranker.main
    python -m ranker.main --output my_submission.csv --top-k 100 --verbose

Pipeline steps
──────────────
1.  Load artifacts (FAISS index, candidate metadata, JD requirements)
2.  Embed the JD → unit-norm query vector
3.  FAISS search → top-K candidate IDs + cosine scores
4.  Honeypot detection → disqualification set
5.  Composite scoring (semantic + skills + experience + behavioral)
6.  Rank → keep top 100, filter disqualified
7.  Generate per-candidate reasoning
8.  Write CSV  (candidate_id, rank, score, reasoning)

Compute constraints (from submission_spec.md §3)
─────────────────────────────────────────────────
• CPU only (no GPU during ranking)
• ≤ 16 GB RAM
• ≤ 5 minutes wall-clock
• No network access during ranking
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import polars as pl
import numpy as np

from .artifacts import load_artifacts
from .config import PATHS, TOP_K_RETRIEVE, TOP_K_FINAL
from .embed import embed_jd
from .reasoning import generate_reasoning
from .scorer import score_candidates


def run(
    output_path: Path | None = None,
    top_k: int = TOP_K_FINAL,
    verbose: bool = True,
) -> Path:
    """
    Full ranking pipeline.

    Parameters
    ----------
    output_path : Path, optional
        Where to write the CSV. Defaults to config.PATHS["output_csv"].
    top_k : int
        Number of candidates in the final CSV (default 100).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    t_total = time.perf_counter()
    out = Path(output_path or PATHS["output_csv"])

    # ── Step 1: Load artifacts ────────────────────────────────────────────────
    _log("═" * 60, verbose)
    _log("PHASE 2 — CANDIDATE RANKING", verbose)
    _log("═" * 60, verbose)

    artifacts = load_artifacts(verbose=verbose)
    index        = artifacts.faiss_index
    cand_ids     = artifacts.candidate_ids
    meta         = artifacts.meta
    jd_req       = artifacts.jd_req

    # ── Step 2: Embed JD ──────────────────────────────────────────────────────
    _log("\n[main] Step 2: Embedding JD …", verbose)
    jd_vec = embed_jd(jd_req)                   # shape (dim,)
    query  = jd_vec.reshape(1, -1)              # FAISS expects (n_queries, dim)

    # ── Step 3: FAISS retrieval ───────────────────────────────────────────────
    _log(f"\n[main] Step 3: FAISS top-{TOP_K_RETRIEVE} retrieval …", verbose)
    t_faiss = time.perf_counter()
    dists, indices = index.search(query, TOP_K_RETRIEVE)
    dists   = dists[0]     # (K,)
    indices = indices[0]   # (K,)

    # Remove FAISS sentinel (-1 for unfilled slots)
    valid_mask = indices >= 0
    dists   = dists[valid_mask]
    indices = indices[valid_mask]

    # Cosine sims from IndexFlatIP are in [-1, 1]; shift to [0, 1]
    dists = (dists + 1.0) / 2.0

    _log(f"[main]   Retrieved {len(indices):,} candidates in "
         f"{time.perf_counter() - t_faiss:.3f}s", verbose)

    # ── Step 4: Honeypot detection ────────────────────────────────────────────
    _log("\n[main] Step 4: Honeypot detection …", verbose)
    t_hp = time.perf_counter()

    # Only run honeypot checks on the retrieved set for speed
    retrieved_ids  = [cand_ids[i] for i in indices]
    retrieved_meta = meta.filter(
        meta["candidate_id"].is_in(retrieved_ids)
    )
    disqualified = set(
        meta.filter(pl.col("honeypot_flag"))["candidate_id"].to_list()
    )

    _log(f"[main]   Disqualified {len(disqualified)} honeypot(s) in "
         f"{time.perf_counter() - t_hp:.3f}s", verbose)

    if len(disqualified) > 0:
        sample = list(disqualified)[:5]
        _log(f"[main]   Sample disqualified IDs: {sample}", verbose)

    # ── Step 5 & 6: Score + Rank ──────────────────────────────────────────────
    _log("\n[main] Steps 5-6: Composite scoring + ranking …", verbose)
    t_score = time.perf_counter()

    ranked = score_candidates(
        faiss_ids=indices,
        faiss_dists=dists,
        candidate_ids=cand_ids,
        meta=meta,
        jd_req=jd_req,
        disqualified=disqualified,
    )

    _log(f"[main]   Scored {len(ranked):,} candidates in "
         f"{time.perf_counter() - t_score:.3f}s", verbose)

    top_n = ranked.head(top_k)
    _log(f"\n[main] Top {len(top_n)} candidates selected.", verbose)
    _log(f"[main]   Score range: "
         f"{top_n['final_score'].min():.4f} – {top_n['final_score'].max():.4f}",
         verbose)

    # ── Step 7: Generate reasoning ────────────────────────────────────────────
    _log(f"\n[main] Step 7: Generating reasoning for top {len(top_n)} …", verbose)
    rows = top_n.to_dicts()
    reasoning_col: list[str] = []
    for row in rows:
        reasoning_col.append(generate_reasoning(row, jd_req))

    # ── Step 8: Write CSV ─────────────────────────────────────────────────────
    _log(f"\n[main] Step 8: Writing CSV → {out}", verbose)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (row, reason) in enumerate(zip(rows, reasoning_col), start=1):
            writer.writerow([
                row["candidate_id"],
                rank,
                f"{row['final_score']:.6f}",
                reason,
            ])

    _log(f"\n[main] Done. {len(rows)} candidates written to {out}", verbose)
    _log(f"[main] Total wall-clock: {time.perf_counter() - t_total:.2f}s", verbose)
    _log("═" * 60, verbose)

    # Sanity checks before returning
    _sanity_check(rows, disqualified, verbose)

    return out


# ── Sanity checks ─────────────────────────────────────────────────────────────

def _sanity_check(
    rows: list[dict],
    disqualified: set[str],
    verbose: bool,
) -> None:
    """Print warnings if submission is at risk of disqualification."""
    _log("\n[sanity] Running submission checks …", verbose)

    # Check 1: No honeypots in top-100
    hp_in_top = [r for r in rows if r["candidate_id"] in disqualified]
    if hp_in_top:
        _log(f"[sanity] ⚠  WARNING: {len(hp_in_top)} disqualified candidates "
             f"leaked into top-100 — check scorer filter!", verbose)
    else:
        _log(f"[sanity] ✓  0 honeypots in top-100", verbose)

    # Check 2: Honeypot rate in top-100 < 10% (submission rule)
    hp_rate_pct = len(hp_in_top) / len(rows) * 100 if rows else 0
    if hp_rate_pct > 10:
        _log(f"[sanity] ✗  DISQUALIFYING: honeypot rate {hp_rate_pct:.1f}% > 10%", verbose)
    else:
        _log(f"[sanity] ✓  Honeypot rate {hp_rate_pct:.1f}% < 10%", verbose)

    # Check 3: Exactly 100 candidates
    if len(rows) != 100:
        _log(f"[sanity] ⚠  Expected 100 candidates, got {len(rows)}", verbose)
    else:
        _log(f"[sanity] ✓  Exactly 100 candidates", verbose)

    # Check 4: Ranks are unique
    ranks = [i + 1 for i in range(len(rows))]
    if len(set(ranks)) == len(rows):
        _log(f"[sanity] ✓  Ranks are unique (1–{len(rows)})", verbose)

    # Check 5: Score is monotone descending
    scores = [r["final_score"] for r in rows]
    is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if is_sorted:
        _log(f"[sanity] ✓  Scores are monotone descending", verbose)
    else:
        _log(f"[sanity] ⚠  Scores are NOT monotone descending!", verbose)

    _log("[sanity] Done.\n", verbose)


# ── Logging helper ────────────────────────────────────────────────────────────

def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redrob Phase-2 Ranker — produces submission.csv"
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: submission.csv in project root)",
    )
    p.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K_FINAL,
        help=f"Number of candidates in output (default: {TOP_K_FINAL})",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_path = run(
        output_path=args.output,
        top_k=args.top_k,
        verbose=not args.quiet,
    )
    print(f"\nSubmission ready: {out_path}")
