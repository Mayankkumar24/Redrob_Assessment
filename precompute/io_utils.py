"""
Step 1.1 — Data Ingestion & Cleaning

Reads the candidate pool (plain .jsonl or gzipped .jsonl.gz), validates each
record against candidate_schema.json, and returns a list of clean Python
dicts. Malformed records are logged and skipped rather than crashing the
whole pipeline — at 100K records we expect the occasional edge case.

Usage:
    from precompute.io_utils import load_candidates
    records, report = load_candidates("data/raw/candidates.jsonl.gz",
                                        "config/candidate_schema.json")
"""

import gzip
import json
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


def _open_any(path: str):
    """Open .jsonl or .jsonl.gz transparently, always as text mode."""
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "rt", encoding="utf-8")


def load_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidates(
    data_path: str,
    schema_path: str,
    max_records: int | None = None,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load and validate candidate records.

    Args:
        data_path: path to .jsonl or .jsonl.gz file (or a plain JSON array,
                    e.g. sample_candidates.json — auto-detected).
        schema_path: path to candidate_schema.json
        max_records: optional cap, useful for quick local testing
        verbose: print progress

    Returns:
        (valid_records, report) where report has counts + a sample of errors.
    """
    schema = load_schema(schema_path)
    validator = Draft7Validator(schema)

    t0 = time.time()
    valid_records: list[dict] = []
    invalid_samples: list[dict] = []
    total = 0
    invalid_count = 0
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    p = Path(data_path)

    # sample_candidates.json ships as a pretty-printed JSON array, not NDJSON.
    # Auto-detect and handle both formats so the same loader works for both
    # the 50-candidate sample and the full 100K NDJSON pool.
    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            raw_iter = json.load(f)
        if not isinstance(raw_iter, list):
            raise ValueError(f"{data_path} is JSON but not a list of records")
        line_iter = raw_iter
    else:
        line_iter = _iter_jsonl(p)

    for raw in line_iter:
        total += 1
        if max_records is not None and total > max_records:
            total -= 1
            break

        try:
            record = raw if isinstance(raw, dict) else json.loads(raw)
        except json.JSONDecodeError as e:
            invalid_count += 1
            if len(invalid_samples) < 10:
                invalid_samples.append({"error": f"JSON decode error: {e}"})
            continue

        if validator.is_valid(record):
            cid = record.get("candidate_id")
            if cid in seen_ids:
                duplicate_ids.append(cid)
            else:
                seen_ids.add(cid)
                valid_records.append(record)
        else:
            invalid_count += 1
            if len(invalid_samples) < 10:
                errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
                msgs = [f"{list(e.path)}: {e.message}" for e in errors[:3]]
                invalid_samples.append(
                    {"candidate_id": record.get("candidate_id", "UNKNOWN"), "errors": msgs}
                )

        if verbose and total % 20000 == 0:
            print(f"  ...processed {total:,} records")

    elapsed = time.time() - t0
    report = {
        "total_seen": total,
        "valid": len(valid_records),
        "invalid": invalid_count,
        "duplicate_ids_dropped": len(duplicate_ids),
        "elapsed_sec": round(elapsed, 2),
        "invalid_samples": invalid_samples,
    }

    if verbose:
        print(f"[io_utils] Loaded {report['valid']:,}/{report['total_seen']:,} valid "
              f"records in {report['elapsed_sec']}s "
              f"({report['invalid']} invalid, {report['duplicate_ids_dropped']} duplicates)")

    return valid_records, report


def _iter_jsonl(path: Path):
    with _open_any(str(path)) as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


if __name__ == "__main__":
    # Quick manual test against the 50-candidate sample
    records, report = load_candidates(
        "data/raw/sample_candidates.json",
        "config/candidate_schema.json",
    )
    print(json.dumps(report, indent=2))
