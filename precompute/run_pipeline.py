"""
run_pipeline.py — Orchestrates Steps 1.1 -> 1.7

This is the single entry point for the offline precompute phase. It runs:

  1.1  Ingestion & validation        (io_utils.py)
  1.1  Flattening to Polars          (flatten.py)
  1.2  Text construction + dedup     (text_builder.py)
  1.3  JD requirements (loaded)      (jd_requirements.py)
  1.4  Disqualifier flags            (disqualifiers.py)
  1.5  Embeddings + production score (embed_candidates.py)  ** needs network **
  1.6  FAISS index                   (build_faiss_index.py)
  1.7  Honeypot detection            (honeypot_detector.py)

...and writes everything Phase 2 (runtime ranking) needs to data/processed/:

  candidates_clean.parquet           - flat enriched dataframe, all candidates
  candidate_embeddings.npy           - (N, 384) float32, L2-normalized
  candidate_ids.npy                  - (N,) ids, same order as embeddings
  candidate_index.faiss              - FAISS IndexFlatIP over the embeddings
  production_experience_scores.json  - {candidate_id: score}

Usage:
    python precompute/run_pipeline.py --input data/raw/candidates.jsonl.gz
    python precompute/run_pipeline.py --input data/raw/sample_candidates.json --skip-embeddings
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from io_utils import load_candidates
from flatten import build_flat_dataframe
from text_builder import build_embedding_texts
from jd_requirements import load_jd_requirements
from disqualifiers import compute_disqualifier_flags
from honeypot_detector import detect_honeypots


def run_pipeline(
    candidates_path: str,
    schema_path: str = "config/candidate_schema.json",
    jd_requirements_path: str = "config/jd_requirements.json",
    output_dir: str = "data/processed",
    max_records: int | None = None,
    skip_embeddings: bool = False,
):
    t_start = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Step 1.1: Ingest & validate ----
    print("=" * 70)
    print("STEP 1.1 — Ingestion & Validation")
    print("=" * 70)
    records, ingest_report = load_candidates(
        candidates_path, schema_path, max_records=max_records
    )
    with open(out / "ingestion_report.json", "w") as f:
        json.dump(ingest_report, f, indent=2)

    # ---- Step 1.1 (cont): Flatten to Polars ----
    print("\n" + "=" * 70)
    print("STEP 1.1 (cont) — Flattening to Polars DataFrame")
    print("=" * 70)
    df = build_flat_dataframe(records)
    print(f"Flattened: {df.shape[0]} rows x {df.shape[1]} columns")

    # ---- Step 1.2: Text construction + dedup ----
    print("\n" + "=" * 70)
    print("STEP 1.2 — Text Construction & Deduplication")
    print("=" * 70)
    embedding_texts = build_embedding_texts(records)
    df = df.with_columns(
        pl.Series("embedding_text", [embedding_texts[cid] for cid in df["candidate_id"]])
    )
    avg_len = sum(len(t) for t in embedding_texts.values()) / len(embedding_texts)
    print(f"Built embedding_text for {len(embedding_texts)} candidates "
          f"(avg {avg_len:.0f} chars)")

    # ---- Step 1.3: JD requirements (just loaded, no computation) ----
    print("\n" + "=" * 70)
    print("STEP 1.3 — JD Requirements (loaded from static config)")
    print("=" * 70)
    jd_requirements = load_jd_requirements(jd_requirements_path)
    print(f"Loaded requirements for role: {jd_requirements['role_title']}")

    # ---- Step 1.4: Disqualifier flags ----
    print("\n" + "=" * 70)
    print("STEP 1.4 — Disqualifier Flag Enrichment")
    print("=" * 70)
    disq_flags = compute_disqualifier_flags(records, jd_requirements)
    disq_df = pl.DataFrame(
        [{"candidate_id": cid, **flags} for cid, flags in disq_flags.items()]
    )
    df = df.join(disq_df, on="candidate_id", how="left")
    n_consulting = sum(1 for f in disq_flags.values() if f["is_consulting_only"])
    n_hopper = sum(1 for f in disq_flags.values() if f["is_job_hopper"])
    n_cvspeech = sum(1 for f in disq_flags.values() if f["cv_speech_only_flag"])
    print(f"Consulting-only: {n_consulting} | Job-hopper: {n_hopper} | "
          f"CV/Speech-only: {n_cvspeech}")

    # ---- Step 1.7: Honeypot detection (runs before 1.5/1.6 - pure rules,
    # no embedding dependency, cheap to run early) ----
    print("\n" + "=" * 70)
    print("STEP 1.7 — Honeypot Detection")
    print("=" * 70)
    honeypot_results = detect_honeypots(records)
    hp_df = pl.DataFrame([
        {
            "candidate_id": cid,
            "honeypot_flag": r["honeypot_flag"],
            "honeypot_reasons": "; ".join(r["honeypot_reasons"]) if r["honeypot_reasons"] else "",
            "contradiction_flag": r["contradiction_flag"],
            "contradiction_reasons": "; ".join(r["contradiction_reasons"]) if r["contradiction_reasons"] else "",
            "skill_duration_penalty_score": r["skill_duration_penalty_score"],
            "soft_contradiction_penalty_score": r["soft_contradiction_penalty_score"],
        }
        for cid, r in honeypot_results.items()
    ])
    df = df.join(hp_df, on="candidate_id", how="left")
    n_hp = sum(1 for r in honeypot_results.values() if r["honeypot_flag"])
    n_contra = sum(1 for r in honeypot_results.values() if r["contradiction_flag"])
    print(f"HARD honeypots flagged: {n_hp} ({100*n_hp/len(records):.2f}%)")
    print(f"SOFT contradictions flagged: {n_contra} ({100*n_contra/len(records):.2f}%)")

    # ---- Step 1.5 + 1.6: Embeddings + FAISS index (network required) ----
    if skip_embeddings:
        print("\n" + "=" * 70)
        print("STEP 1.5/1.6 — SKIPPED (--skip-embeddings flag set)")
        print("=" * 70)
        print("Run embed_candidates.py and build_faiss_index.py separately "
              "on a machine with internet access to huggingface.co.")
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("production_experience_score"))
    else:
        print("\n" + "=" * 70)
        print("STEP 1.5 — Candidate Embedding Generation (needs network)")
        print("=" * 70)
        from embed_candidates import (
            _load_model, generate_embeddings,
            compute_production_experience_scores, save_artifacts,
        )
        model = _load_model()
        embeddings, candidate_ids = generate_embeddings(embedding_texts, model=model)
        production_scores = compute_production_experience_scores(
            embedding_texts, embeddings, candidate_ids, model=model
        )
        save_artifacts(embeddings, candidate_ids, production_scores, output_dir)

        prod_df = pl.DataFrame(
            [{"candidate_id": cid, "production_experience_score": score}
             for cid, score in production_scores.items()]
        )
        df = df.join(prod_df, on="candidate_id", how="left")

        print("\n" + "=" * 70)
        print("STEP 1.6 — FAISS Index Construction")
        print("=" * 70)
        from build_faiss_index import run as build_faiss_run
        build_faiss_run(
            embeddings_path=str(out / "candidate_embeddings.npy"),
            output_path=str(out / "candidate_index.faiss"),
        )

    # ---- Save final enriched dataframe ----
    print("\n" + "=" * 70)
    print("SAVING FINAL ARTIFACTS")
    print("=" * 70)
    parquet_path = out / "candidates_clean.parquet"
    df.write_parquet(parquet_path)
    print(f"Saved {parquet_path} ({df.shape[0]} rows x {df.shape[1]} cols)")

    elapsed = time.time() - t_start
    print(f"\nPipeline complete in {elapsed:.1f}s")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Steps 1.1-1.7 of the offline precompute pipeline")
    parser.add_argument("--input", default="data/raw/sample_candidates.json",
                         help="Path to candidates.jsonl.gz, candidates.jsonl, or sample_candidates.json")
    parser.add_argument("--schema", default="config/candidate_schema.json")
    parser.add_argument("--jd-requirements", default="config/jd_requirements.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-records", type=int, default=None,
                         help="Cap records for quick local testing")
    parser.add_argument("--skip-embeddings", action="store_true",
                         help="Skip Steps 1.5/1.6 (useful in network-restricted environments)")
    args = parser.parse_args()

    run_pipeline(
        candidates_path=args.input,
        schema_path=args.schema,
        jd_requirements_path=args.jd_requirements,
        output_dir=args.output_dir,
        max_records=args.max_records,
        skip_embeddings=args.skip_embeddings,
    )
