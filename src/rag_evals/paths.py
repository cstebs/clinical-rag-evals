"""Canonical filesystem locations.

Everything in the project resolves paths through here so that a single
``RAG_EVALS_ROOT`` override relocates the whole working tree (useful in CI and
in tests, which point it at a tmpdir).
"""

from __future__ import annotations

import os
from pathlib import Path

# src/rag_evals/paths.py -> src/rag_evals -> src -> repo root
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

ROOT = Path(os.environ.get("RAG_EVALS_ROOT", _DEFAULT_ROOT))

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "manifest.json"

EVALS = ROOT / "evals"
GOLD = EVALS / "gold.jsonl"
HUMAN_LABELS = EVALS / "human_labels.jsonl"

CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"

CACHE = ROOT / ".cache"
JUDGE_CACHE = ROOT / ".judge_cache"
MODELS = ROOT / ".models"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for d in (DATA, RAW, PROCESSED, EVALS, RESULTS, CACHE, JUDGE_CACHE, MODELS):
        d.mkdir(parents=True, exist_ok=True)
