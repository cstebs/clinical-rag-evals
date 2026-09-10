"""config -> full eval -> one results row.

Execution order matters and is not arbitrary:

1. Chunk and index. Free, deterministic, no credentials.
2. Retrieve for every gold question and score retrieval. Still free, still
   deterministic. **If this stage looks wrong, stop here** - generation metrics
   computed on top of broken retrieval measure nothing but the model's prior
   knowledge.
3. Generate, then judge. Costs money, so it runs last and only if asked.

Stages 1 and 2 are why ``--no-generate`` exists and why the whole chunking and
retrieval sweep can be run for $0.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import gold as gold_mod
from . import metrics, paths
from .chunk import chunk_corpus
from .config import RunConfig
from .generate import generate_batch
from .ingest import Manifest
from .judge import judge_batch
from .llm import LLMClient, LLMUnavailable
from .retrieve import Retriever


class ConfigUnavailable(RuntimeError):
    """This environment cannot honour the config as written.

    Distinct from a crash: the config is valid, but something it requires (model
    credentials, for instance) is not present. Callers skip the run rather than
    emitting a results row that claims a feature which did not actually execute.
    """


@dataclass
class RunResult:
    run_id: str
    payload: dict[str, Any]

    def save(self) -> None:
        paths.RESULTS.mkdir(parents=True, exist_ok=True)
        out = paths.RESULTS / f"{self.run_id}.json"
        out.write_text(
            json.dumps(self.payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {out}")


def usable_doc_ids(manifest: Manifest) -> list[str]:
    """Documents that parsed cleanly.

    Excluding low-yield (scanned) documents is a corpus decision, not a silent
    convenience: a document whose text never extracted cannot be retrieved by
    any strategy, so leaving it in would depress every row of the table by the
    same amount and hide the real differences. The count of exclusions is
    recorded in every run so the decision stays visible.
    """
    return [d.doc_id for d in manifest.docs if d.parse_ok]


def execute(
    cfg: RunConfig,
    generate: bool = True,
    stub_llm=None,
    limit_questions: int | None = None,
) -> RunResult:
    t0 = time.monotonic()
    manifest = Manifest.load()
    doc_ids = usable_doc_ids(manifest)
    excluded = len(manifest.docs) - len(doc_ids)

    questions = gold_mod.load()
    if limit_questions:
        questions = questions[:limit_questions]
    if not questions:
        raise SystemExit("evals/gold.jsonl is empty; author the gold set first")

    # Fail before doing any work rather than emitting a row that claims a
    # feature was active when it silently was not.
    if cfg.retrieval.query_rewrite and stub_llm is None and not LLMClient.credentials_available():
        raise ConfigUnavailable(
            "query rewriting requires model credentials. "
            "Set ANTHROPIC_API_KEY (or run `ant auth login`), or pass --stub-llm."
        )

    print(f"[{cfg.run_id}] {len(doc_ids)} documents, {len(questions)} questions")

    # -- 1. chunk -------------------------------------------------------
    t = time.monotonic()
    chunks, chunk_stats = chunk_corpus(
        doc_ids,
        strategy=cfg.chunking.strategy,
        target_tokens=cfg.chunking.target_tokens,
        overlap_tokens=cfg.chunking.overlap_tokens,
    )
    t_chunk = time.monotonic() - t
    print(
        f"  chunked -> {chunk_stats['n_chunks']:,} chunks "
        f"(median {chunk_stats['median_tokens']} tok, "
        f"{chunk_stats['tokens_lost_pct']}% of tokens beyond the encoder window)"
    )

    # -- 2. retrieve ----------------------------------------------------
    llm: LLMClient | None = None
    if cfg.retrieval.query_rewrite or generate:
        llm = LLMClient(model=cfg.generation.model, stub=stub_llm)

    t = time.monotonic()
    retriever = Retriever(chunks, cfg.retrieval, llm=llm)
    t_index = time.monotonic() - t

    t = time.monotonic()
    per_question: list[dict[str, Any]] = []
    hit_lists = []
    from tqdm import tqdm

    for q in tqdm(questions, desc="  retrieving"):
        hits = retriever.search(q.question, cfg.retrieval.top_k)
        hit_lists.append(hits)
        per_question.append(
            {
                "qid": q.qid,
                "question": q.question,
                "question_type": q.question_type,
                "gold_docs": list(q.source_docs),
                "retrieved": [h.to_dict() for h in hits],
                "retrieved_docs": metrics.doc_ranking([h.doc_id for h in hits]),
            }
        )
    t_retrieve = time.monotonic() - t

    retrieval_scores = metrics.score_retrieval(
        [([h.doc_id for h in hits], q.source_docs) for q, hits in zip(questions, hit_lists)]
    )
    print(f"  retrieval: {retrieval_scores.flat()}")

    payload: dict[str, Any] = {
        "run_id": cfg.run_id,
        "description": cfg.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "corpus": {
            "n_docs_selected": len(manifest.docs),
            "n_docs_used": len(doc_ids),
            "n_docs_excluded_low_yield": excluded,
            "manifest_created_at": manifest.created_at,
        },
        "chunking_stats": chunk_stats,
        "gold": gold_mod.summarize(questions),
        "retrieval": retrieval_scores.flat()
        | {
            "n_scored": retrieval_scores.n_scored,
            "n_negative_excluded": retrieval_scores.n_skipped_negative,
        },
        "generation": None,
        "usage": None,
        "timings_seconds": {
            "chunk": round(t_chunk, 2),
            "index": round(t_index, 2),
            "retrieve": round(t_retrieve, 2),
        },
        "per_question": per_question,
    }

    # -- 3. generate + judge --------------------------------------------
    if generate and cfg.generation.enabled:
        assert llm is not None
        try:
            t = time.monotonic()
            answers = generate_batch(
                llm,
                [q.question for q in questions],
                hit_lists,
                cfg.generation.max_context_chars,
            )
            t_gen = time.monotonic() - t

            t = time.monotonic()
            verdicts = judge_batch(llm, questions, answers, cfg.generation.max_context_chars)
            t_judge = time.monotonic() - t

            records = []
            for q, a, v, row in zip(questions, answers, verdicts, per_question):
                row["answer"] = a.text
                row["abstained"] = v.abstained
                row["judge"] = v.to_dict()
                records.append(
                    {
                        "is_negative": q.is_negative,
                        "abstained": v.abstained,
                        "grounded": v.grounded,
                        "correct": v.correct,
                    }
                )

            gen_scores = metrics.score_generation(records)
            payload["generation"] = gen_scores.flat() | {
                "n_answerable": gen_scores.n_answerable,
                "n_negative": gen_scores.n_negative,
                "false_abstentions": gen_scores.false_abstentions,
                "confabulations": gen_scores.confabulations,
                "judge_parse_failures": sum(1 for v in verdicts if v.parse_failed),
            }
            payload["timings_seconds"]["generate"] = round(t_gen, 2)
            payload["timings_seconds"]["judge"] = round(t_judge, 2)
            print(f"  generation: {gen_scores.flat()}")
        except LLMUnavailable as exc:
            payload["generation"] = {"skipped": str(exc)}
            print(f"  generation skipped: {exc}")

    if llm is not None:
        payload["usage"] = llm.usage.summary()

    payload["timings_seconds"]["total"] = round(time.monotonic() - t0, 2)
    return RunResult(cfg.run_id, payload)
