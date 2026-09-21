"""Command line interface.

    rag-evals ingest            fetch + parse the FDA corpus, write the manifest
    rag-evals validate-gold     check the gold set against the parsed corpus
    rag-evals chunk-stats       compare chunking strategies without retrieving
    rag-evals run CONFIG        one config -> one results row
    rag-evals sweep             run every config in configs/
    rag-evals report            regenerate the results table
    rag-evals label             hand-label a run, with the judge verdict hidden
    rag-evals validate-judge    judge vs. human labels: agreement and kappa
    rag-evals estimate-cost     budget a sweep before spending on it
    rag-evals prompts           list / show / diff / audit versioned system prompts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import gold as gold_mod
from . import metrics, paths, report
from .config import RunConfig, isolation_warning
from .llm import LLMClient, estimate_cost
from .run import ConfigUnavailable


def _stub(call):
    """Deterministic offline stand-in for a model call.

    Exists so the generation and judging paths can be exercised end to end with
    no credentials. It abstains on everything and grades everything as grounded,
    which produces *structurally valid but substantively meaningless* numbers -
    so any run made with it is stamped ``stub_llm: true`` in its record and must
    never be reported as a result.
    """
    if "grading" in call.system:
        return json.dumps(
            {
                "grounded": True,
                "correct": False,
                "abstained": True,
                "unsupported_claims": [],
                "reasoning": "stub judge: no model call was made",
            }
        )
    return json.dumps(
        {"abstained": True, "answer": "INSUFFICIENT_CONTEXT", "cited_passages": []}
    )


# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    from . import ingest

    ingest.run(
        limit=args.limit,
        delay=args.delay,
        force_fetch=args.force,
        force_parse=args.force_parse,
        skip_fetch=args.skip_fetch,
    )
    return 0


def cmd_validate_gold(args: argparse.Namespace) -> int:
    from .ingest import Manifest

    questions = gold_mod.load()
    known = None
    if paths.MANIFEST.exists():
        known = {d.doc_id for d in Manifest.load().docs}

    issues = gold_mod.validate(questions, known)
    summary = gold_mod.summarize(questions)
    print(json.dumps(summary, indent=2))

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    for i in issues:
        print(f"  {i.severity.upper():7} {i.qid}: {i.message}")
    print(f"\n{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


def cmd_chunk_stats(args: argparse.Namespace) -> int:
    from .chunk import chunk_corpus
    from .ingest import Manifest
    from .run import usable_doc_ids

    doc_ids = usable_doc_ids(Manifest.load())
    rows = []
    for strategy in ("fixed", "recursive", "section"):
        for size in args.sizes:
            _chunks, stats = chunk_corpus(doc_ids, strategy, size, args.overlap)
            rows.append(stats)
            print(
                f"{strategy:10} size={size:5}  chunks={stats['n_chunks']:6,}  "
                f"median={stats['median_tokens']:5}  p95={stats['p95_tokens']:5}  "
                f"over-limit={stats['chunks_over_limit_pct']:5.1f}%  "
                f"tokens-lost={stats['tokens_lost_pct']:5.2f}%  "
                f"sections={stats['sections_detected']:,}"
            )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .run import execute

    cfg = RunConfig.load(args.config)
    if cfg.baseline:
        base_path = paths.CONFIGS / f"{cfg.baseline}.yaml"
        if base_path.exists():
            warn = isolation_warning(cfg, RunConfig.load(base_path))
            if warn:
                print(f"WARNING: {warn}", file=sys.stderr)

    try:
        result = execute(
            cfg,
            generate=not args.no_generate,
            stub_llm=_stub if args.stub_llm else None,
            limit_questions=args.limit_questions,
        )
    except ConfigUnavailable as exc:
        print(f"cannot run {cfg.run_id}: {exc}", file=sys.stderr)
        return 2
    if args.stub_llm:
        result.payload["stub_llm"] = True
    result.save()
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .run import execute

    configs = sorted(paths.CONFIGS.glob("*.yaml"))
    if not configs:
        print("no configs found in configs/", file=sys.stderr)
        return 1
    skipped: list[str] = []
    for path in configs:
        cfg = RunConfig.load(path)
        print(f"\n=== {cfg.run_id}: {cfg.description} ===")
        try:
            result = execute(
                cfg,
                generate=not args.no_generate,
                stub_llm=_stub if args.stub_llm else None,
                limit_questions=args.limit_questions,
            )
        except ConfigUnavailable as exc:
            # A config this environment cannot honour is skipped loudly and
            # leaves no results file behind. A missing row is honest; a row
            # claiming a feature that silently did not run is not. The rest of
            # the sweep continues.
            print(f"SKIPPED {cfg.run_id}: {exc}", file=sys.stderr)
            skipped.append(cfg.run_id)
            continue
        if args.stub_llm:
            result.payload["stub_llm"] = True
        result.save()

    if skipped:
        print(f"\n{len(skipped)} config(s) skipped: {', '.join(skipped)}", file=sys.stderr)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    # Injected content nests under the README's own "Results" heading, so it
    # starts a level deeper than the standalone report.
    text = report.build_report(heading_level=3 if args.inject else 2)
    if args.inject:
        n = report.inject(Path(args.inject), text)
        print(f"spliced {n:,} characters into {args.inject}")
    elif args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


class _Quit(Exception):
    """The labeller asked to stop. Everything decided so far is already saved."""


def _ask(question: str, extra: str = "") -> bool:
    """Ask one yes/no question. Raises _Quit on 'q'."""
    hint = "[y/n" + (f"/{extra}" if extra else "") + "/q=save and quit]"
    while True:
        try:
            raw = input(f"  {question} {hint} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            raise _Quit from None
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        if raw in ("q", "quit"):
            raise _Quit
        print("    please answer y or n (or q to save and quit)")


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def cmd_label(args: argparse.Namespace) -> int:
    """Hand-label a run's answers without seeing the judge's verdicts.

    The point of this command is *structural blindness*. Judge validation is
    only meaningful if the human decided independently, and a procedure that
    relies on the labeller not scrolling down to the verdict in a JSON file is a
    promise, not a control. So:

    * the judge's verdict for the item is never printed, and
    * the reference answer is withheld until after ``grounded`` is committed,
      because groundedness is a question about the retrieved passages alone and
      seeing the reference first is exactly the anchor that would corrupt it.

    Labels are appended one at a time, so quitting never loses prior work, and a
    resumed session continues the same stratified sample rather than drawing a
    new one.
    """
    from .chunk import chunk_corpus
    from .config import ChunkingConfig
    from .judge import HumanLabel, HumanLabelError, append_human_label, load_human_labels, select_for_labelling

    run_path = paths.RESULTS / f"{args.run_id}.json"
    if not run_path.exists():
        print(f"no results file for run {args.run_id!r}", file=sys.stderr)
        return 1
    run = json.loads(run_path.read_text(encoding="utf-8"))

    if run.get("stub_llm"):
        print(
            f"run {args.run_id!r} was produced with --stub-llm; there is nothing real "
            "to label. Re-run it with credentials first.",
            file=sys.stderr,
        )
        return 1
    if not any(r.get("judge") for r in run.get("per_question", [])):
        print(
            f"run {args.run_id!r} has no generated answers. Re-run it without "
            "--no-generate before labelling.",
            file=sys.stderr,
        )
        return 1

    # Passage text is not stored in the run record (it would multiply its size
    # for data that is exactly reproducible). Re-derive it from the run's own
    # chunking config: same config, same chunk ids, same text.
    ch = ChunkingConfig(**run["config"]["chunking"])
    print(
        f"rebuilding passages from the run's chunking config "
        f"({ch.strategy}/{ch.target_tokens}) ..."
    )
    from .ingest import Manifest
    from .run import usable_doc_ids

    chunks, _ = chunk_corpus(
        usable_doc_ids(Manifest.load()),
        strategy=ch.strategy,
        target_tokens=ch.target_tokens,
        overlap_tokens=ch.overlap_tokens,
    )
    text_of = {c.chunk_id: c for c in chunks}

    # Reference answers live in the gold set, not the run record.
    expected = {q.qid: q.expected_answer for q in gold_mod.load()}

    rows = {r["qid"]: r for r in run["per_question"]}
    try:
        existing = load_human_labels(paths.HUMAN_LABELS) if paths.HUMAN_LABELS.exists() else []
    except HumanLabelError as e:
        print(f"human_labels.jsonl has a bad line, fix it before labelling:\n  {e}", file=sys.stderr)
        return 1
    already = {l.qid for l in existing if l.run_id == args.run_id}
    queue = select_for_labelling(
        [(r["qid"], r["question_type"]) for r in run["per_question"]],
        already,
        args.n,
        seed=args.seed,
    )

    if not queue:
        print(f"nothing left to label for {args.run_id!r} ({len(already)} already done).")
        return 0

    print(_rule("="))
    print(f"Labelling {len(queue)} items from run {args.run_id!r}"
          f"{f' ({len(already)} already labelled)' if already else ''}.")
    print("The judge's verdict is hidden. The reference answer appears only after")
    print("you have committed a groundedness call. 'q' saves and quits at any point.")
    print(_rule("="))

    done = 0
    for i, qid in enumerate(queue, start=1):
        row = rows[qid]
        try:
            print()
            print(_rule("="))
            print(f"[{i}/{len(queue)}]  {qid}  ({row['question_type']})")
            print(_rule("="))
            print(f"\nQUESTION\n  {row['question']}\n")

            print(f"RETRIEVED PASSAGES (top {args.passages} of {len(row['retrieved'])})")
            for p in row["retrieved"][: args.passages]:
                chunk = text_of.get(p["chunk_id"])
                body = chunk.text if chunk else "(passage text unavailable)"
                if args.truncate and len(body) > args.truncate:
                    body = body[: args.truncate].rstrip() + " [...]"
                print(f"\n  [{p['rank']}] {p['doc_id']}  pages {p['pages'][0]}-{p['pages'][1]}")
                for line in body.splitlines():
                    print(f"      {line}")
            print()

            print("SYSTEM ANSWER")
            answer = row.get("answer") or "(empty)"
            for line in answer.splitlines():
                print(f"  {line}")
            print()

            abstained = _ask("Did it DECLINE to answer (not merely hedge)?")

            if abstained:
                grounded = True
                print("  -> grounded = true automatically (an abstention asserts nothing),")
                print("     which is the same rule the judge is given.")
            else:
                grounded = _ask("Is EVERY claim supported by those passages alone?")

            # Only now is the reference revealed.
            is_negative = row["question_type"] == "negative"
            print()
            print("REFERENCE ANSWER")
            if is_negative:
                print("  (none: this question is NOT answerable from the corpus;")
                print("   the correct behaviour is to decline)")
                correct = abstained
                print(f"  -> correct = {str(correct).lower()} (equals your abstention call;")
                print("     correctness is undefined without a reference answer)")
            else:
                for line in (expected.get(qid) or "(not recorded)").splitlines():
                    print(f"  {line}")
                print()
                correct = _ask("Does the system answer match it in substance?")

            try:
                note = input("  note (optional, enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                raise _Quit from None

            append_human_label(
                paths.HUMAN_LABELS,
                HumanLabel(
                    qid=qid,
                    run_id=args.run_id,
                    grounded=grounded,
                    correct=correct,
                    abstained=abstained,
                    note=note,
                ),
            )
            done += 1
        except _Quit:
            break

    print()
    print(_rule("="))
    print(f"saved {done} label(s) to {paths.HUMAN_LABELS}  ({len(already) + done} total "
          f"for {args.run_id!r})")
    if done:
        print(f"next: rag-evals validate-judge --run-id {args.run_id}")
    return 0


def cmd_validate_judge(args: argparse.Namespace) -> int:
    """Compare judge verdicts against hand labels and publish the agreement."""
    from .judge import HumanLabelError, load_human_labels

    if not paths.HUMAN_LABELS.exists():
        print(f"no human labels at {paths.HUMAN_LABELS}", file=sys.stderr)
        return 1
    try:
        labels = load_human_labels(paths.HUMAN_LABELS)
    except HumanLabelError as e:
        print(f"human_labels.jsonl has a bad line, fix it and rerun:\n  {e}", file=sys.stderr)
        return 1
    if not labels:
        print("human_labels.jsonl is empty", file=sys.stderr)
        return 1

    run_ids = {l.run_id for l in labels}
    if len(run_ids) > 1 and not args.run_id:
        print(f"labels span multiple runs {sorted(run_ids)}; pass --run-id", file=sys.stderr)
        return 1
    run_id = args.run_id or next(iter(run_ids))

    run_path = paths.RESULTS / f"{run_id}.json"
    if not run_path.exists():
        print(f"no results file for run {run_id!r}", file=sys.stderr)
        return 1
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("stub_llm"):
        print(f"run {run_id!r} was produced with --stub-llm; agreement would be meaningless",
              file=sys.stderr)
        return 1

    by_qid = {r["qid"]: r for r in run["per_question"]}
    out: dict[str, object] = {"run_id": run_id, "n_labels": len(labels)}
    disagreements: list[dict[str, object]] = []

    for field in ("grounded", "correct", "abstained"):
        judge_vals, human_vals = [], []
        for label in labels:
            row = by_qid.get(label.qid)
            if not row or not row.get("judge"):
                continue
            # Correctness is undefined for a question with no reference answer.
            # Comparing it on negatives would mix a coin-flip into the one
            # number the project is judged on.
            if field == "correct" and row["question_type"] == "negative":
                continue
            j = bool(row["judge"][field])
            h = bool(getattr(label, field))
            judge_vals.append(j)
            human_vals.append(h)
            if j != h:
                disagreements.append(
                    {
                        "qid": label.qid,
                        "field": field,
                        "judge": j,
                        "human": h,
                        "judge_reasoning": row["judge"]["reasoning"],
                        "human_note": label.note,
                    }
                )
        agreement = metrics.cohen_kappa(judge_vals, human_vals)
        out[field] = {
            "n": agreement.n,
            "raw_agreement": round(agreement.raw_agreement, 4),
            "cohen_kappa": round(agreement.cohen_kappa, 4),
            "judge_lenient": agreement.judge_yes_human_no,
            "judge_harsh": agreement.judge_no_human_yes,
        }
        print(f"{field:11} {agreement.summary()}")

    out["disagreements"] = disagreements
    dest = paths.RESULTS / f"judge-validation-{run_id}.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {dest}  ({len(disagreements)} disagreements recorded)")
    return 0


def cmd_estimate_cost(args: argparse.Namespace) -> int:
    n_q = args.questions
    if n_q is None:
        n_q = len(gold_mod.load()) if paths.GOLD.exists() else 80
    n_c = args.configs
    if n_c is None:
        n_c = max(1, len(list(paths.CONFIGS.glob("*.yaml"))))
    for batch in (True, False):
        est = estimate_cost(n_q, n_c, batch=batch)
        tag = "batch (50%)" if batch else "sync"
        print(f"{tag:14} {est['calls']:5,} calls  ~${est['estimated_cost_usd']:.2f}")
    print(f"\n(for {n_q} questions x {n_c} configs x 2 calls each; cache hits are free)")
    return 0


# ---------------------------------------------------------------------------


def cmd_prompts(args: argparse.Namespace) -> int:
    """Versioned prompts: list, show, diff, audit.

    ``audit`` is the one that matters. It exits non-zero if any committed run
    record references prompt text that no longer matches the file on disk, so
    an in-place edit of a published version cannot pass unnoticed.
    """
    import difflib

    from . import prompts as prompts_mod

    if args.action == "list":
        used = prompts_mod.usage_by_results()
        for p in prompts_mod.all_prompts():
            runs = used.get((p.name, p.version), [])
            print(
                f"{p.name:9} {p.version:5} {p.status:8} {p.sha256[:12]}  "
                f"created {p.meta.get('created', '?')}  runs: {', '.join(runs) or '-'}"
            )
            print(f"{'':36}{p.meta.get('changes', '')}")
        return 0

    if args.action == "show":
        if not args.name or len(args.versions) != 1:
            print("usage: rag-evals prompts show NAME VERSION", file=sys.stderr)
            return 2
        p = prompts_mod.load(args.name, args.versions[0])
        print(f"# {p.name}/{p.version}  status={p.status}  sha256={p.sha256}")
        for k, v in p.meta.items():
            print(f"# {k}: {v}")
        print()
        print(p.text)
        return 0

    if args.action == "diff":
        if not args.name or len(args.versions) != 2:
            print("usage: rag-evals prompts diff NAME OLD_VERSION NEW_VERSION", file=sys.stderr)
            return 2
        a = prompts_mod.load(args.name, args.versions[0])
        b = prompts_mod.load(args.name, args.versions[1])
        print(f"# {b.name}/{b.version} changes: {b.meta.get('changes', '')}")
        for line in difflib.unified_diff(
            a.text.splitlines(), b.text.splitlines(),
            fromfile=f"{a.name}/{a.version}", tofile=f"{b.name}/{b.version}", lineterm="",
        ):
            print(line)
        return 0

    if args.action == "audit":
        problems, notes = prompts_mod.audit()
        for n in notes:
            print(f"note: {n}")
        for pr in problems:
            print(f"PROBLEM: {pr}", file=sys.stderr)
        if problems:
            print(f"\n{len(problems)} problem(s). Published prompt versions are immutable; "
                  "put changed text in a new version.", file=sys.stderr)
            return 1
        print(f"\nprompt audit clean: {len(prompts_mod.all_prompts())} version(s) on disk, "
              f"every run record and config consistent with them.")
        return 0

    return 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag-evals", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="fetch + parse the FDA corpus")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--delay", type=float, default=None,
                   help="extra politeness delay; cannot go below robots.txt Crawl-Delay")
    s.add_argument("--force", action="store_true", help="re-download existing PDFs")
    s.add_argument("--force-parse", action="store_true")
    s.add_argument("--skip-fetch", action="store_true", help="re-parse what is already on disk")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("validate-gold", help="check the gold set against the corpus")
    s.set_defaults(func=cmd_validate_gold)

    s = sub.add_parser("chunk-stats", help="compare chunkers without retrieving")
    s.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    s.add_argument("--overlap", type=int, default=0)
    s.add_argument("--json", default=None)
    s.set_defaults(func=cmd_chunk_stats)

    s = sub.add_parser("run", help="execute one config")
    s.add_argument("config")
    s.add_argument("--no-generate", action="store_true",
                   help="retrieval metrics only; costs nothing")
    s.add_argument("--stub-llm", action="store_true",
                   help="offline stand-in for model calls; results are not reportable")
    s.add_argument("--limit-questions", type=int, default=None)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="execute every config in configs/")
    s.add_argument("--no-generate", action="store_true")
    s.add_argument("--stub-llm", action="store_true")
    s.add_argument("--limit-questions", type=int, default=None)
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("report", help="regenerate the results table")
    s.add_argument("--out", default=None)
    s.add_argument("--inject", default=None,
                   help="splice the report into a file between the RESULTS markers")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("label", help="hand-label a run's answers, judge verdict hidden")
    s.add_argument("--run-id", required=True)
    s.add_argument("--n", type=int, default=40, help="how many items to label (default 40)")
    s.add_argument("--passages", type=int, default=5,
                   help="how many retrieved passages to show (default 5)")
    s.add_argument("--truncate", type=int, default=1200,
                   help="max characters per passage; 0 for full text")
    s.add_argument("--seed", type=int, default=11)
    s.set_defaults(func=cmd_label)

    s = sub.add_parser("validate-judge", help="judge vs. human labels")
    s.add_argument("--run-id", default=None)
    s.set_defaults(func=cmd_validate_judge)

    s = sub.add_parser("estimate-cost", help="budget a sweep")
    s.add_argument("--questions", type=int, default=None)
    s.add_argument("--configs", type=int, default=None)
    s.set_defaults(func=cmd_estimate_cost)

    s = sub.add_parser("prompts", help="list, show, diff, or audit versioned system prompts")
    s.add_argument("action", choices=["list", "show", "diff", "audit"])
    s.add_argument("name", nargs="?", help="generate | judge | rewrite (show, diff)")
    s.add_argument("versions", nargs="*", help="one version for show, two for diff")
    s.set_defaults(func=cmd_prompts)

    # A Windows console defaults to cp1252 and raises on any non-Latin-1
    # character in generated output. Reports are UTF-8; make the stream match.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = p.parse_args(argv)
    paths.ensure_dirs()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
