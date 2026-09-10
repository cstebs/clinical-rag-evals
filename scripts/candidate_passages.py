"""Surface high-signal passages to author gold questions from.

This is the tool behind ``evals/gold.jsonl``. Rather than inventing questions
and hoping the corpus answers them, the process runs the other way: pull
concrete, checkable statements out of the real parsed text, then write the
question each one answers. That ordering is what makes every gold item
answerable by construction, and it is why ``rag-evals validate-gold`` can verify
each quote verbatim afterwards.

Selection heuristics - a passage is a candidate when it is:

* long enough to be specific but short enough to quote (80-320 characters),
* not front matter (contact blocks, table of contents, dotted leaders), and
* carrying a regulatory signal: a modal ("should", "must", "recommends"), a
  definition ("means", "is defined as"), or a concrete quantity.

Usage::

    python scripts/candidate_passages.py --docs 12 --per-doc 6
    python scripts/candidate_passages.py --doc-id fda-71336
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_evals import paths  # noqa: E402

SIGNAL = re.compile(
    r"\b(should|must|recommends?|is defined as|means that|means the|"
    r"required to|no later than|within \d+|at least \d+|\d+\s?(days?|months?|years?|percent|%))\b",
    re.I,
)

NOISE = re.compile(
    r"\.{4,}|_{4,}|https?://|@|Rockville|Silver Spring|Fishers Lane|"
    r"Additional copies|Tel\b|Fax|Phone:|Table of Contents|"
    r"^\s*[IVXLC]+\.\s*$|Contains Nonbinding",
    re.I,
)

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def passages(doc_id: str, min_page: int = 3) -> list[tuple[int, str]]:
    payload = json.loads((paths.PROCESSED / f"{doc_id}.json").read_text(encoding="utf-8"))
    by_page: dict[int, list[str]] = {}
    for line in payload["lines"]:
        by_page.setdefault(line["page"], []).append(line["t"])

    out: list[tuple[int, str]] = []
    for page, lines in sorted(by_page.items()):
        if page < min_page:
            continue
        text = " ".join(lines)
        for sent in SENTENCE.split(text):
            sent = sent.strip()
            if not (80 <= len(sent) <= 320):
                continue
            if NOISE.search(sent):
                continue
            if not SIGNAL.search(sent):
                continue
            out.append((page, sent))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=10, help="number of documents to sample")
    ap.add_argument("--per-doc", type=int, default=5)
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        available = sorted(p.stem for p in paths.PROCESSED.glob("*.json"))
        doc_ids = available[args.offset : args.offset + args.docs]

    rng = random.Random(args.seed)
    for doc_id in doc_ids:
        try:
            cands = passages(doc_id)
        except FileNotFoundError:
            continue
        if not cands:
            continue
        payload = json.loads((paths.PROCESSED / f"{doc_id}.json").read_text(encoding="utf-8"))
        print("=" * 100)
        print(f"{doc_id} | {payload['title'][:88]}")
        picks = cands if len(cands) <= args.per_doc else rng.sample(cands, args.per_doc)
        for page, sent in sorted(picks):
            print(f"  p{page}: {sent}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
