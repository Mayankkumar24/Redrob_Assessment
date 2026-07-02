"""
ranker/main.py
───────────────
Phase 2 complete pipeline.

Steps:
  1.  Load top_170.json
  2.  Load candidates_clean.parquet
  3.  Filter honeypots
  4.  Re-scale production scores within pool → [0, 1]
  5.  Embed JD (bge-base-en-v1.5, BGE query prefix, 512-token short JD)
  6.  Extract embeddings for top-170 → compute semantic scores
  7.  Score each candidate (prod + semantic + skills + behavioral + penalty)
  8.  Sort descending; tie-break candidate_id ascending
  9.  Take top-100
  10. Stretch final scores → [0.05, 0.98] for better separation
  11. Generate per-candidate reasoning
  12. Write submission.csv

Usage:
    python -m ranker.main
    python -m ranker.main --output my_submission.csv --quiet
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import polars as pl

from .config import PATHS, TOP_K_FINAL
from .embed import embed_jd, load_candidate_semantic_scores
from .scorer import score_candidate
from .reasoning import get_reasoning


def run(
    output_path: Path | None = None,
    top_k: int = TOP_K_FINAL,
    verbose: bool = True,
) -> Path:

    t_total = time.perf_counter()
    out     = Path(output_path or PATHS["output_csv"])

    _log("=" * 60, verbose)
    _log("PHASE 2 — CANDIDATE RANKING", verbose)
    _log("=" * 60, verbose)

    # ── Step 1: Load top_170.json ─────────────────────────────────────────────
    _log("\n[1/12] Loading top_170.json …", verbose)
    top_170_path = PATHS["top_170"]
    if not top_170_path.exists():
        raise FileNotFoundError(
            f"top_170.json not found: {top_170_path}\n"
            "Generate from production_experience_scores.json:\n"
            "  Sort desc → top 170 → save as ranker/top_170.json\n"
            '  Format: [{"candidate_id": "CAND_...", '
            '"production_experience_score": 0.95}, ...]'
        )
    with open(top_170_path, encoding="utf-8") as f:
        top_170: list[dict] = json.load(f)
    _log(f"  {len(top_170)} candidates loaded", verbose)
    top_170_reasoning: dict[str, str] = {
        c["candidate_id"]: c.get("reasoning", "")
        for c in top_170
    }

    # ── Step 2: Load candidate metadata ──────────────────────────────────────
    _log("\n[2/12] Loading candidates_clean.parquet …", verbose)
    meta_path = PATHS["candidate_meta"]
    if not meta_path.exists():
        raise FileNotFoundError(f"Parquet not found: {meta_path}")
    meta = pl.read_parquet(meta_path)
    _log(f"  {meta.height:,} rows, {meta.width} columns", verbose)

    # ── Step 3: Filter honeypots ──────────────────────────────────────────────
    _log("\n[3/12] Filtering honeypots …", verbose)
    honeypot_ids: set[str] = set()
    if "honeypot_flag" in meta.columns:
        honeypot_ids = set(
            meta.filter(pl.col("honeypot_flag") == True)["candidate_id"].to_list()
        )
    _log(f"  {len(honeypot_ids)} honeypot IDs found in parquet", verbose)

    top_170_clean = [c for c in top_170 if c["candidate_id"] not in honeypot_ids]
    removed = len(top_170) - len(top_170_clean)
    _log(f"  Removed {removed} → {len(top_170_clean)} candidates remain", verbose)

    if len(top_170_clean) < top_k:
        _log(
            f"  WARNING: Only {len(top_170_clean)} candidates remain "
            f"(need {top_k}). Increase top_170 pool size.",
            verbose
        )

    # ── Step 4: Re-scale production scores within pool ────────────────────────
    _log("\n[4/12] Re-scaling production scores …", verbose)
    scores = [float(c["production_experience_score"]) for c in top_170_clean]
    s_min, s_max = min(scores), max(scores)
    for c in top_170_clean:
        c["prod_rescaled"] = (
            (float(c["production_experience_score"]) - s_min) / (s_max - s_min + 1e-9)
        )
    _log(f"  Original range: [{s_min:.4f}, {s_max:.4f}] → rescaled to [0, 1]", verbose)

    # ── Step 5: Embed JD ──────────────────────────────────────────────────────
    _log("\n[5/12] Embedding JD (bge-base-en-v1.5) …", verbose)
    t_emb  = time.perf_counter()
    jd_vec = embed_jd()
    _log(f"  JD embedded in {time.perf_counter() - t_emb:.2f}s  dim={jd_vec.shape[0]}", verbose)

    # ── Step 6: Semantic scores for top-170 ──────────────────────────────────
    _log("\n[6/12] Computing semantic scores for top-170 …", verbose)
    t_sem = time.perf_counter()
    pool_ids = [c["candidate_id"] for c in top_170_clean]
    semantic_scores: dict[str, float] = load_candidate_semantic_scores(
        pool_ids, jd_vec, verbose=verbose
    )
    _log(f"  Done in {time.perf_counter() - t_sem:.2f}s", verbose)

    # ── Step 7: Build metadata lookup ─────────────────────────────────────────
    _log("\n[7/12] Building metadata lookup …", verbose)
    meta_pool = meta.filter(pl.col("candidate_id").is_in(pool_ids))
    meta_dict: dict[str, dict] = {
        row["candidate_id"]: row
        for row in meta_pool.to_dicts()
    }
    _log(f"  {len(meta_dict)} candidates found in parquet", verbose)

    # ── Step 8: Score each candidate ──────────────────────────────────────────
    _log("\n[8/12] Scoring candidates …", verbose)
    t_score  = time.perf_counter()
    results: list[dict] = []

    for cand_entry in top_170_clean:
        cid = cand_entry["candidate_id"]
        row = meta_dict.get(cid)
        if row is None:
            _log(f"  WARNING: {cid} missing from parquet — skipping", verbose)
            continue
        sem_score = semantic_scores.get(cid, 0.5)
        result    = score_candidate(cand_entry, row, sem_score)
        results.append(result)

    _log(
        f"  Scored {len(results)} candidates in "
        f"{time.perf_counter() - t_score:.2f}s",
        verbose
    )

    # ── Step 9: Sort + top-K ──────────────────────────────────────────────────
    _log(f"\n[9/12] Sorting → top {top_k} …", verbose)
    results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    top_n = results[:top_k]
    _log(
        f"  Score range: [{top_n[-1]['final_score']:.4f}, {top_n[0]['final_score']:.4f}]",
        verbose
    )

    # ── Step 10: Stretch scores ───────────────────────────────────────────────
    _log("\n[10/12] Stretching scores …", verbose)
    fmin = top_n[-1]["final_score"]
    fmax = top_n[0]["final_score"]
    if fmax > fmin:
        for r in top_n:
            r["final_score"] = round(
                ((r["final_score"] - fmin) / (fmax - fmin)) * 0.93 + 0.05,
                6
            )
    _log(
        f"  After stretch: [{top_n[-1]['final_score']:.4f}, {top_n[0]['final_score']:.4f}]",
        verbose
    )

    # ── Step 11: Reasoning ────────────────────────────────────────────────────
    _log(f"\n[11/12] Reading reasoning from top_170 …", verbose)
    for r in top_n:
        r["reasoning"] = get_reasoning(r["candidate_id"], top_170_reasoning)

    # ── Step 12: Write CSV ────────────────────────────────────────────────────
    _log(f"\n[12/12] Writing CSV → {out}", verbose)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top_n, start=1):
            writer.writerow([
                r["candidate_id"],
                rank,
                f"{r['final_score']:.6f}",
                r["reasoning"],
            ])

    elapsed = time.perf_counter() - t_total
    _log(f"\nDone. {len(top_n)} candidates in {elapsed:.2f}s", verbose)
    _log("=" * 60, verbose)

    _sanity_check(top_n, honeypot_ids, verbose)
    return out


# ── Sanity checks ─────────────────────────────────────────────────────────────

def _sanity_check(top_n: list[dict], honeypot_ids: set[str], verbose: bool) -> None:
    _log("\n[sanity] Running checks …", verbose)

    hp_in = [r for r in top_n if r["candidate_id"] in honeypot_ids]
    _log(f"  {'✓' if not hp_in else '✗'} Honeypots in top-{len(top_n)}: {len(hp_in)}", verbose)

    _log(f"  {'✓' if len(top_n) == 100 else '✗'} Count: {len(top_n)}", verbose)

    scores = [r["final_score"] for r in top_n]
    mono   = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    _log(f"  {'✓' if mono else '✗'} Scores monotone descending", verbose)

    hp_rate = len(hp_in) / len(top_n) * 100 if top_n else 0
    _log(
        f"  {'✓' if hp_rate <= 10 else '✗'} Honeypot rate: {hp_rate:.1f}%",
        verbose
    )
    _log("[sanity] Done.\n", verbose)


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Redrob Phase 2 Ranker")
    p.add_argument("--output", "-o", type=Path, default=None)
    p.add_argument("--top-k", "-k", type=int, default=TOP_K_FINAL)
    p.add_argument("--quiet", "-q", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    out_path = run(
        output_path=args.output,
        top_k=args.top_k,
        verbose=not args.quiet,
    )
    print(f"\nSubmission ready: {out_path}")