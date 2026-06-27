"""
ranker/
───────
Phase-2 ranking module for the Redrob Hackathon.

Submodules
──────────
config      – paths, weights, thresholds (edit here to tune)
artifacts   – load precomputed Phase-1 artifacts
embed       – JD embedding via BAAI/bge-small-en-v1.5
honeypot    – hard disqualification rules
behavioral  – redrob_signals → behavioral multiplier
scorer      – composite scoring (semantic + skills + exp + behavioral)
reasoning   – per-candidate 1-2 sentence reasoning strings
main        – CLI entry point → submission.csv

Quick start
───────────
    python -m ranker.main
    python -m ranker.main --output my_sub.csv --verbose
"""

__version__ = "2.0.0"
