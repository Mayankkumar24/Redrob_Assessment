"""
review_honeypots.py — Manual Profile Reviewer for Honeypot-Flagged Candidates

After running run_pipeline.py, this script lets you interactively browse the
full raw profile of every candidate flagged as a HARD honeypot or a SOFT
contradiction.

Two modes:
  1. [DEFAULT] Read from data/processed/candidates_clean.parquet
     (pipeline must have been run first — flags are already there)

  2. [--live] Re-run honeypot detection on the fly from the raw input file
     (useful if the parquet has not been built yet, e.g. --skip-embeddings run)

Usage examples:
    # Browse all HARD flagged candidates interactively (paginated)
    python precompute/review_honeypots.py

    # Review a specific candidate by ID
    python precompute/review_honeypots.py --id cand_100

    # Show all SOFT contradiction candidates too
    python precompute/review_honeypots.py --type soft

    # Show BOTH hard and soft
    python precompute/review_honeypots.py --type all

    # Re-detect live without needing the parquet
    python precompute/review_honeypots.py --live --input data/raw/sample_candidates.json

    # Dump flagged candidates to a JSON file for offline review
    python precompute/review_honeypots.py --type all --export flagged_profiles.json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 72
THIN_SEP  = "-" * 72


def _color(text: str, code: str) -> str:
    try:
        return f"\033[{code}m{text}\033[0m"
    except Exception:
        return text


def RED(t):    return _color(str(t), "91")
def YELLOW(t): return _color(str(t), "93")
def GREEN(t):  return _color(str(t), "92")
def CYAN(t):   return _color(str(t), "96")
def BOLD(t):   return _color(str(t), "1")
def DIM(t):    return _color(str(t), "2")


def _print_section(title: str, data: Any) -> None:
    print(f"\n{CYAN(title)}")
    print(DIM(THIN_SEP[:40]))
    if data is None:
        print(DIM("  (not present)"))
    elif isinstance(data, list):
        if not data:
            print(DIM("  (empty list)"))
        for item in data:
            if isinstance(item, dict):
                for k, v in item.items():
                    print(f"  {BOLD(k)}: {v}")
                print()
            else:
                print(f"  - {item}")
    elif isinstance(data, dict):
        for k, v in data.items():
            print(f"  {BOLD(k)}: {v}")
    else:
        print(f"  {data}")


def _print_flag_box(flag_type: str, reasons: list[str]) -> None:
    if flag_type == "HARD":
        label  = RED("  HARD HONEYPOT FLAG  [excluded from ranking]")
        border = RED(SEPARATOR)
    else:
        label  = YELLOW("  SOFT CONTRADICTION FLAG  [down-weighted in scoring]")
        border = YELLOW(SEPARATOR)

    print(f"\n{border}")
    print(label)
    print(border)
    for r in reasons:
        prefix = RED("  x ") if flag_type == "HARD" else YELLOW("  ~ ")
        print(f"{prefix}{r}")
    print(border)


def print_full_profile(record: dict, hp_result: dict) -> None:
    cid      = record.get("candidate_id", "UNKNOWN")
    profile  = record.get("profile", {})
    skills   = record.get("skills", [])
    edu      = record.get("education", [])
    career   = record.get("career_history", [])
    certs    = record.get("certifications", [])
    signals  = record.get("redrob_signals", {})
    projects = record.get("projects", [])
    achievem = record.get("achievements", [])
    langs    = record.get("languages", [])

    is_hard = hp_result["honeypot_flag"]
    is_soft = hp_result["contradiction_flag"]

    print(f"\n\n{SEPARATOR}")
    id_line = f"  CANDIDATE PROFILE:  {BOLD(cid)}"
    if is_hard:
        id_line += f"   {RED('[HARD HONEYPOT]')}"
    elif is_soft:
        id_line += f"   {YELLOW('[SOFT CONTRADICTION]')}"
    else:
        id_line += f"   {GREEN('[CLEAN]')}"
    print(id_line)
    print(SEPARATOR)

    if is_hard:
        _print_flag_box("HARD", hp_result["honeypot_reasons"])
    if is_soft:
        _print_flag_box("SOFT", hp_result["contradiction_reasons"])

    # Core profile
    _print_section("PROFILE SUMMARY", profile)

    # Skills
    if skills:
        print(f"\n{CYAN('SKILLS')}")
        print(DIM(THIN_SEP[:40]))
        for s in skills:
            prof   = s.get("proficiency", "?")
            months = s.get("duration_months", "?")
            name   = s.get("name", "?")
            is_bad = (prof == "expert" and months == 0)
            line   = f"  - {name:<30}  proficiency={prof:<12}  duration={months} months"
            print(RED(line) if is_bad else line)

    # Education
    if edu:
        print(f"\n{CYAN('EDUCATION')}")
        print(DIM(THIN_SEP[:40]))
        for e in edu:
            start = e.get("start_year", "?")
            end   = e.get("end_year", "?")
            flag  = RED("  <- END < START!") if (
                isinstance(start, int) and isinstance(end, int) and end < start
            ) else ""
            print(f"  - {e.get('institution','?')} | {e.get('degree','?')} in {e.get('field_of_study','?')}")
            print(f"    {start} -> {end}{flag}")

    # Career history
    if career:
        print(f"\n{CYAN('CAREER HISTORY')}")
        print(DIM(THIN_SEP[:40]))
        for j in career:
            current_tag = GREEN(" [CURRENT]") if j.get("is_current") else ""
            print(f"  - {BOLD(j.get('company','?'))} -- {j.get('title','?')}{current_tag}")
            print(f"    {j.get('start_date','?')} -> {j.get('end_date','null')}  |  {j.get('duration_months','?')} months claimed")
            resps = j.get("responsibilities", [])
            if resps:
                preview = resps[0] if resps else ""
                print(f"    Responsibilities: {textwrap.shorten(str(preview), width=68)}")
            print()

    # Certifications
    if certs:
        print(f"\n{CYAN('CERTIFICATIONS')}")
        print(DIM(THIN_SEP[:40]))
        for c in certs:
            yr = c.get("year") or c.get("issue_year", "?")
            print(f"  - {c.get('name','?')}  [{yr}]  -- {c.get('issuer','?')}")

    # Projects
    if projects:
        print(f"\n{CYAN('PROJECTS')}")
        print(DIM(THIN_SEP[:40]))
        for p in projects:
            print(f"  - {p.get('name','?')}: {textwrap.shorten(str(p.get('description','')), width=68)}")

    # Achievements
    if achievem:
        print(f"\n{CYAN('ACHIEVEMENTS')}")
        print(DIM(THIN_SEP[:40]))
        for a in achievem:
            print(f"  - {a}")

    # Languages
    if langs:
        print(f"\n{CYAN('LANGUAGES')}")
        print(DIM(THIN_SEP[:40]))
        for l in langs:
            print(f"  - {l.get('language','?')} -- {l.get('proficiency','?')}")

    # Redrob Signals
    if signals:
        print(f"\n{CYAN('REDROB SIGNALS')}")
        print(DIM(THIN_SEP[:40]))
        for k, v in signals.items():
            print(f"  {BOLD(k)}: {v}")

    print(f"\n{DIM(SEPARATOR)}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_raw_records(input_path: str) -> dict[str, dict]:
    import gzip
    p = Path(input_path)
    records: dict[str, dict] = {}

    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for rec in arr:
            cid = rec.get("candidate_id")
            if cid:
                records[cid] = rec
    else:
        opener = (gzip.open(p, "rt", encoding="utf-8")
                  if p.suffix == ".gz"
                  else open(p, "rt", encoding="utf-8"))
        with opener as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cid = rec.get("candidate_id")
                    if cid:
                        records[cid] = rec
                except json.JSONDecodeError:
                    pass
    return records


def _load_flags_from_parquet(parquet_path: str) -> dict[str, dict]:
    try:
        import polars as pl
    except ImportError:
        print("ERROR: polars not installed. Use --live mode instead.", file=sys.stderr)
        sys.exit(1)

    df = pl.read_parquet(parquet_path)
    required = {"candidate_id", "honeypot_flag", "honeypot_reasons",
                "contradiction_flag", "contradiction_reasons"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Parquet missing columns: {missing}\n"
              f"Run the pipeline first or use --live mode.", file=sys.stderr)
        sys.exit(1)

    flags: dict[str, dict] = {}
    for row in df.select(list(required)).to_dicts():
        cid = row["candidate_id"]
        flags[cid] = {
            "candidate_id":          cid,
            "honeypot_flag":         bool(row["honeypot_flag"]),
            "honeypot_reasons":      [r for r in (row["honeypot_reasons"] or "").split("; ") if r],
            "contradiction_flag":    bool(row["contradiction_flag"]),
            "contradiction_reasons": [r for r in (row["contradiction_reasons"] or "").split("; ") if r],
        }
    return flags


def _detect_live(input_path: str, schema_path: str) -> tuple[dict[str, dict], dict[str, dict]]:
    sys.path.insert(0, str(Path(__file__).parent))
    from io_utils import load_candidates
    from honeypot_detector import detect_honeypots

    print(f"[live] Loading candidates from {input_path} ...")
    records, _ = load_candidates(input_path, schema_path, verbose=True)
    raw_by_id = {r["candidate_id"]: r for r in records}

    print("[live] Running honeypot detection ...")
    flags = detect_honeypots(records)
    return raw_by_id, flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually review honeypot-flagged candidate profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",   default="data/raw/candidates.jsonl",
                        help="Raw candidates file (.json/.jsonl/.jsonl.gz)")
    parser.add_argument("--schema",  default="config/candidate_schema.json",
                        help="Schema file (only used in --live mode)")
    parser.add_argument("--parquet", default="data/processed/candidates_clean.parquet",
                        help="Parquet written by run_pipeline.py (default mode)")
    parser.add_argument("--live",    action="store_true",
                        help="Re-run detection from scratch instead of reading parquet")
    parser.add_argument("--id",      default=None,
                        help="Review a specific candidate_id (e.g. cand_100)")
    parser.add_argument("--type",    choices=["hard", "soft", "all"], default="hard",
                        help="Which flags to show: hard|soft|all  (default: hard)")
    parser.add_argument("--export",  default=None,
                        help="Export flagged profiles+reasons to this JSON file and exit")
    parser.add_argument("--no-pager", action="store_true",
                        help="Print all profiles without interactive pause")
    args = parser.parse_args()

    # Resolve paths relative to repo root so the script works from any cwd
    repo_root    = Path(__file__).parent.parent
    input_path   = Path(args.input)   if Path(args.input).is_absolute()   else repo_root / args.input
    parquet_path = Path(args.parquet) if Path(args.parquet).is_absolute() else repo_root / args.parquet
    schema_path  = Path(args.schema)  if Path(args.schema).is_absolute()  else repo_root / args.schema

    # ── 1. Get flags ────────────────────────────────────────────────────────
    if args.live:
        raw_by_id, flags = _detect_live(str(input_path), str(schema_path))
    else:
        if not parquet_path.exists():
            print(f"ERROR: Parquet not found at {parquet_path}\n"
                  f"Run run_pipeline.py first, or use --live flag.", file=sys.stderr)
            sys.exit(1)
        flags = _load_flags_from_parquet(str(parquet_path))
        print(f"[parquet] Loaded flags for {len(flags):,} candidates.")
        print(f"[raw]     Loading raw profiles from {input_path} ...")
        raw_by_id = _load_raw_records(str(input_path))
        print(f"[raw]     Loaded {len(raw_by_id):,} raw profiles.")

    # ── 2. Filter ───────────────────────────────────────────────────────────
    if args.id:
        if args.id not in flags:
            if args.id in raw_by_id:
                sys.path.insert(0, str(Path(__file__).parent))
                from honeypot_detector import evaluate_candidate
                flags[args.id] = evaluate_candidate(raw_by_id[args.id])
            else:
                print(f"ERROR: '{args.id}' not found.", file=sys.stderr)
                sys.exit(1)
        target_ids = [args.id]
    else:
        def _matches(r: dict) -> bool:
            if args.type == "hard":
                return r["honeypot_flag"]
            if args.type == "soft":
                return r["contradiction_flag"] and not r["honeypot_flag"]
            return r["honeypot_flag"] or r["contradiction_flag"]
        target_ids = [cid for cid, r in flags.items() if _matches(r)]

    # ── 3. Summary ──────────────────────────────────────────────────────────
    n_hard  = sum(1 for r in flags.values() if r["honeypot_flag"])
    n_soft  = sum(1 for r in flags.values() if r["contradiction_flag"])
    n_total = len(flags)
    print(f"\n{SEPARATOR}")
    print(BOLD("  HONEYPOT REVIEW TOOL"))
    print(SEPARATOR)
    print(f"  Total candidates  : {n_total:,}")
    print(f"  HARD honeypots    : {RED(str(n_hard))}  ({100*n_hard/max(n_total,1):.2f}%)")
    print(f"  SOFT contradictions: {YELLOW(str(n_soft))}  ({100*n_soft/max(n_total,1):.2f}%)")
    print(f"  Showing ({args.type}): {BOLD(str(len(target_ids)))} candidate(s)")
    print(f"{SEPARATOR}\n")

    if not target_ids:
        print(GREEN("No flagged candidates for this filter. All clean!"))
        return

    # ── 4. Export mode ──────────────────────────────────────────────────────
    if args.export:
        export_data = []
        for cid in target_ids:
            export_data.append({
                "candidate_id":          cid,
                "honeypot_flag":         flags[cid]["honeypot_flag"],
                "honeypot_reasons":      flags[cid]["honeypot_reasons"],
                "contradiction_flag":    flags[cid]["contradiction_flag"],
                "contradiction_reasons": flags[cid]["contradiction_reasons"],
                "full_profile":          raw_by_id.get(cid, {"_error": "not found"}),
            })
        out_path = Path(args.export)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(export_data)} flagged profiles to {out_path.resolve()}")
        return

    # ── 5. Interactive review ───────────────────────────────────────────────
    for i, cid in enumerate(target_ids, 1):
        hp_result   = flags[cid]
        raw_profile = raw_by_id.get(cid)

        if raw_profile is None:
            print(f"\n[{i}/{len(target_ids)}] {RED(cid)}: raw profile not in input file.")
            all_reasons = hp_result["honeypot_reasons"] + hp_result["contradiction_reasons"]
            for r in all_reasons:
                print(f"  - {r}")
        else:
            print(f"\n[{i}/{len(target_ids)}]", end="")
            print_full_profile(raw_profile, hp_result)

        if not args.no_pager and i < len(target_ids):
            try:
                ans = input(BOLD("Press Enter for next | 'q' to quit: ")).strip().lower()
                if ans == "q":
                    print("\nExiting. Goodbye!")
                    break
            except (EOFError, KeyboardInterrupt):
                break

    print(f"\n{GREEN('Review complete.')}")


if __name__ == "__main__":
    main()
