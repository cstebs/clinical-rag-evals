"""Turn committed run records into the results table and failure analysis.

The README is the deliverable (SPEC section 8.7). This module generates the
parts of it that must never drift from the data: the results table, the deltas
against the baseline, and the failure breakdown. Regenerating is a command, not
a copy-paste, so a stale table is not a failure mode this project can have.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import paths

METRIC_COLUMNS = [
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("mrr", "MRR"),
    ("ndcg@10", "nDCG@10"),
]

GEN_COLUMNS = [
    ("groundedness", "Grounded"),
    ("correctness", "Correct"),
    ("abstention_acc", "Abstain"),
]


def load_runs(results_dir: Path | None = None) -> list[dict[str, Any]]:
    results_dir = results_dir or paths.RESULTS
    runs = []
    for p in sorted(results_dir.glob("*.json")):
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    runs.sort(key=lambda r: r["run_id"])
    return runs


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _label(run: dict[str, Any]) -> str:
    c = run["config"]
    ch, rt = c["chunking"], c["retrieval"]
    bits = [f"{ch['strategy']}/{ch['target_tokens']}"]
    if ch["overlap_tokens"]:
        bits.append(f"ov{ch['overlap_tokens']}")
    bits.append(rt["index"])
    if rt["rerank"]:
        bits.append("+rerank")
    if rt["query_rewrite"]:
        bits.append("+rewrite")
    return " ".join(bits)


def results_table(runs: Iterable[dict[str, Any]]) -> str:
    runs = list(runs)
    if not runs:
        return "_No runs recorded yet._"

    has_gen = any(isinstance(r.get("generation"), dict) and "groundedness" in r["generation"] for r in runs)

    head = ["Run", "Configuration"] + [c[1] for c in METRIC_COLUMNS]
    if has_gen:
        head += [c[1] for c in GEN_COLUMNS]
    lines = ["| " + " | ".join(head) + " |"]
    lines.append("|" + "|".join(["---"] * len(head)) + "|")

    for r in runs:
        row = [f"`{r['run_id']}`", _label(r)]
        row += [_fmt(r["retrieval"].get(k)) for k, _ in METRIC_COLUMNS]
        if has_gen:
            gen = r.get("generation") or {}
            row += [_fmt(gen.get(k)) for k, _ in GEN_COLUMNS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def delta_table(runs: Iterable[dict[str, Any]]) -> str:
    """What each run changed, and what that change bought.

    This is the table the whole project exists to produce. A run with no
    declared baseline is skipped rather than compared against an arbitrary
    neighbour - an unattributed delta is worse than no delta.
    """
    runs = list(runs)
    by_id = {r["run_id"]: r for r in runs}
    # Plain ASCII on purpose: this text is printed to a terminal as well as
    # written to the README, and a Windows console defaulting to cp1252 cannot
    # encode a Greek delta or an arrow.
    lines = [
        "| Run | Changed from baseline | d R@5 | d MRR | d nDCG@10 |",
        "|---|---|---|---|---|",
    ]
    any_row = False

    for r in runs:
        base_id = r["config"].get("baseline")
        if not base_id or base_id not in by_id:
            continue
        base = by_id[base_id]
        changed = _changed_fields(base["config"], r["config"])
        deltas = []
        for key in ("recall@5", "mrr", "ndcg@10"):
            a = base["retrieval"].get(key)
            b = r["retrieval"].get(key)
            deltas.append(f"{b - a:+.3f}" if a is not None and b is not None else "-")
        lines.append(
            f"| `{r['run_id']}` | {changed or '(none)'} | " + " | ".join(deltas) + " |"
        )
        any_row = True

    return "\n".join(lines) if any_row else "_No runs declare a baseline yet._"


def _changed_fields(a: dict[str, Any], b: dict[str, Any]) -> str:
    out = []
    for section in ("chunking", "retrieval", "generation"):
        for k in set(a.get(section, {})) | set(b.get(section, {})):
            av, bv = a.get(section, {}).get(k), b.get(section, {}).get(k)
            if av != bv:
                out.append(f"`{k}`: {av} -> {bv}")
    return ", ".join(sorted(out))


def failure_analysis(run: dict[str, Any], top_n: int = 8, heading_level: int = 3) -> str:
    """Where a run loses, broken out by question type.

    Aggregate metrics hide the interesting part. A system can hold a respectable
    mean recall while failing every multi-hop question, and that is a completely
    different engineering problem from failing uniformly.
    """
    rows = run.get("per_question", [])
    if not rows:
        return "_No per-question records._"

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)

    out = [f"{'#' * heading_level} `{run['run_id']}`: failures by question type\n"]
    out.append("| Type | n | Retrieval hit@5 | Grounded | Correct | Abstained |")
    out.append("|---|---|---|---|---|---|")

    for qtype in ("fact", "multi_hop", "near_miss", "negative"):
        items = by_type.get(qtype, [])
        if not items:
            continue
        if qtype == "negative":
            hit = "-"
        else:
            hits = [
                1.0 if set(r["retrieved_docs"][:5]) & set(r["gold_docs"]) else 0.0
                for r in items
                if r["gold_docs"]
            ]
            hit = f"{sum(hits)/len(hits):.2f}" if hits else "-"
        judged = [r for r in items if r.get("judge")]
        grounded = (
            f"{sum(1 for r in judged if r['judge']['grounded'])/len(judged):.2f}"
            if judged
            else "-"
        )
        correct = (
            f"{sum(1 for r in judged if r['judge']['correct'])/len(judged):.2f}"
            if judged
            else "-"
        )
        abst = (
            f"{sum(1 for r in judged if r['judge']['abstained'])/len(judged):.2f}"
            if judged
            else "-"
        )
        out.append(f"| {qtype} | {len(items)} | {hit} | {grounded} | {correct} | {abst} |")

    misses = [
        r
        for r in rows
        if r["gold_docs"] and not (set(r["retrieved_docs"][:5]) & set(r["gold_docs"]))
    ]
    if misses:
        out.append(f"\n**{len(misses)} questions retrieved nothing relevant in the top 5.**\n")
        for r in misses[:top_n]:
            out.append(f"- `{r['qid']}` ({r['question_type']}): {r['question']}")
            out.append(f"  - expected: {', '.join(r['gold_docs'])}")
            out.append(f"  - got: {', '.join(r['retrieved_docs'][:3]) or '(nothing)'}")

    confab = [
        r
        for r in rows
        if r["question_type"] == "negative" and r.get("judge") and not r["judge"]["abstained"]
    ]
    if confab:
        out.append(
            f"\n**{len(confab)} confabulations**, answered a question the corpus cannot answer:\n"
        )
        for r in confab[:top_n]:
            out.append(f"- `{r['qid']}`: {r['question']}")
            out.append(f"  - answered: {str(r.get('answer',''))[:180]}")
            out.append(f"  - judge: {r['judge']['reasoning'][:220]}")

    return "\n".join(out)


def corpus_summary() -> str:
    if not paths.MANIFEST.exists():
        return "_No manifest; run `rag-evals ingest`._"
    m = json.loads(paths.MANIFEST.read_text(encoding="utf-8"))
    docs = m["docs"]
    ok = [d for d in docs if d.get("parse_ok")]
    pages = sum(d.get("n_pages") or 0 for d in ok)
    chars = sum(d.get("n_chars") or 0 for d in ok)
    centers = Counter(d["center"] for d in ok)
    lines = [
        f"- **Documents selected:** {len(docs)} (from {m['selection']['index_rows_seen']:,} in the FDA index)",
        f"- **Usable after parsing:** {len(ok)} ({len(docs)-len(ok)} excluded for low text yield)",
        f"- **Pages:** {pages:,}  **Characters:** {chars:,}",
        f"- **Issuing centers:** " + ", ".join(f"{k} ({v})" for k, v in centers.most_common()),
        f"- **Index fetched:** {m['created_at'][:10]}",
    ]
    return "\n".join(lines)


def build_report(results_dir: Path | None = None, heading_level: int = 2) -> str:
    """Render the generated block.

    ``heading_level`` exists because this text has two homes. On its own - piped
    to stdout, or written with ``--out`` - its sections are top level. Spliced
    into the README it lands *inside* that file's own "Results" section, so it
    has to start one level down, or the document carries two "Results" headings
    at the same depth and an outline that misdescribes its own structure.
    """
    h = "#" * heading_level
    runs = load_runs(results_dir)
    parts = [
        f"{h} Corpus\n",
        corpus_summary(),
        f"\n\n{h} Retrieval results\n",
        results_table(runs),
        f"\n\n{h} What each change bought\n",
        delta_table(runs),
    ]
    if runs:
        best = max(
            runs, key=lambda r: r["retrieval"].get("ndcg@10") or 0.0
        )
        parts += [
            f"\n\n{h} Failure analysis\n",
            failure_analysis(best, heading_level=heading_level + 1),
        ]
    return "\n".join(parts)


BEGIN_MARKER = "<!-- RESULTS:BEGIN -->"
END_MARKER = "<!-- RESULTS:END -->"


def inject(target: Path, body: str) -> int:
    """Splice generated content into a file between the RESULTS markers.

    The README's numbers are therefore never typed by hand, which removes the
    most common way a results table goes quietly stale: someone tunes a config,
    re-runs, and forgets to update the prose.
    """
    text = target.read_text(encoding="utf-8")
    start = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise SystemExit(f"{target} is missing the RESULTS markers")
    new = (
        text[: start + len(BEGIN_MARKER)]
        + "\n<!-- generated by `rag-evals report --inject README.md`; do not edit by hand -->\n\n"
        + body
        + "\n\n"
        + text[end:]
    )
    target.write_text(new, encoding="utf-8")
    return len(body)
