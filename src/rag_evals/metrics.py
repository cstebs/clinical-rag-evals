"""Retrieval metrics, generation metrics, and judge-agreement statistics.

Everything here is deterministic and model-free. That is the point: the
retrieval half of this harness can be run, re-run, and trusted without spending
a cent or waiting on an API, which is why SPEC section 8 orders it before any
generation work.

The graded unit
---------------
Relevance is graded at **document** granularity, not passage offsets. This is a
deliberate design decision and the single most important one in the file.

If gold labels were pinned to character offsets in a specific passage, then
changing the chunking strategy would change which spans exist, and therefore
change what "correct" means. Every row of a chunking sweep would be scored
against a subtly different rubric, and the deltas - the entire product - would
be meaningless. Grading on document ids keeps the target fixed while the
pipeline underneath it varies.

The gold set still stores verbatim ``source_quotes``. They are used to give the
judge and the human labeller something concrete to check against, and to prove
a question is actually answerable from the corpus. They are never used for
retrieval scoring.

Chunk-budget recall
-------------------
``recall@k`` here means: take the top *k chunks*, collapse them to the set of
distinct documents they came from, and ask what fraction of the gold documents
that set covers. k is a chunk budget because chunks are what actually occupy the
generator's context window. Reporting recall over the top-k *documents* instead
would flatter every configuration, since it silently grants a deeper read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_KS = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def doc_ranking(retrieved_doc_ids: Sequence[str]) -> list[str]:
    """Collapse a chunk ranking to a document ranking, keeping best position."""
    seen: set[str] = set()
    out: list[str] = []
    for d in retrieved_doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(retrieved_doc_ids: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return float("nan")  # negative cases have no retrieval target
    got = set(retrieved_doc_ids[:k])
    return len(got & gold_set) / len(gold_set)


def reciprocal_rank(retrieved_doc_ids: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return float("nan")
    for i, d in enumerate(doc_ranking(retrieved_doc_ids), start=1):
        if d in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_doc_ids: Sequence[str], gold: Iterable[str], k: int = 10) -> float:
    """Binary-relevance nDCG over the deduplicated document ranking.

    Binary because the gold set records "this document contains the answer" and
    nothing finer. Inventing graded relevance would mean inventing labels.
    """
    gold_set = set(gold)
    if not gold_set:
        return float("nan")
    ranking = doc_ranking(retrieved_doc_ids)[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranking, start=1) if d in gold_set)
    ideal_n = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


@dataclass
class RetrievalScores:
    recall: dict[int, float]
    mrr: float
    ndcg10: float
    n_scored: int
    n_skipped_negative: int

    def flat(self) -> dict[str, float]:
        out = {f"recall@{k}": v for k, v in sorted(self.recall.items())}
        out["mrr"] = self.mrr
        out["ndcg@10"] = self.ndcg10
        return {k: round(v, 4) for k, v in out.items()}


def score_retrieval(
    per_question: Sequence[tuple[Sequence[str], Sequence[str]]],
    ks: Sequence[int] = DEFAULT_KS,
) -> RetrievalScores:
    """Aggregate over questions.

    ``per_question`` is a sequence of ``(retrieved_doc_ids, gold_doc_ids)``.
    Questions with an empty gold set - the negative cases - are excluded from
    retrieval scoring entirely and counted separately. Scoring them as 0 would
    punish a system for correctly having nothing to find; scoring them as 1
    would inflate every number. They are graded on abstention instead.
    """
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    rrs: list[float] = []
    ndcgs: list[float] = []
    skipped = 0

    for retrieved, gold in per_question:
        if not gold:
            skipped += 1
            continue
        for k in ks:
            recalls[k].append(recall_at_k(retrieved, gold, k))
        rrs.append(reciprocal_rank(retrieved, gold))
        ndcgs.append(ndcg_at_k(retrieved, gold, 10))

    return RetrievalScores(
        recall={k: _mean(v) for k, v in recalls.items()},
        mrr=_mean(rrs),
        ndcg10=_mean(ndcgs),
        n_scored=len(rrs),
        n_skipped_negative=skipped,
    )


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


@dataclass
class GenerationScores:
    groundedness: float
    correctness: float
    abstention_accuracy: float
    n_answerable: int
    n_negative: int
    false_abstentions: int
    confabulations: int

    def flat(self) -> dict[str, float]:
        return {
            "groundedness": round(self.groundedness, 4),
            "correctness": round(self.correctness, 4),
            "abstention_acc": round(self.abstention_accuracy, 4),
        }


def score_generation(records: Sequence[dict]) -> GenerationScores:
    """Aggregate judged answers.

    Each record needs: ``is_negative`` (bool), ``abstained`` (bool), and, for
    answerable questions, ``grounded`` and ``correct`` (bools from the judge).

    Abstention accuracy is measured over *all* questions, not just the negative
    ones. A system that abstains on everything scores 100% on the negatives and
    must be penalised somewhere - here it is penalised by the false abstentions
    on answerable questions, which drag the same number down.
    """
    answerable = [r for r in records if not r["is_negative"]]
    negative = [r for r in records if r["is_negative"]]

    grounded = [bool(r.get("grounded")) for r in answerable if not r.get("abstained")]
    correct = [bool(r.get("correct")) for r in answerable]

    correct_abstention = sum(1 for r in negative if r.get("abstained"))
    correct_answering = sum(1 for r in answerable if not r.get("abstained"))
    denom = len(records) or 1

    return GenerationScores(
        groundedness=_mean([float(g) for g in grounded]),
        correctness=_mean([float(c) for c in correct]),
        abstention_accuracy=(correct_abstention + correct_answering) / denom,
        n_answerable=len(answerable),
        n_negative=len(negative),
        false_abstentions=sum(1 for r in answerable if r.get("abstained")),
        confabulations=sum(1 for r in negative if not r.get("abstained")),
    )


# ---------------------------------------------------------------------------
# judge validation
# ---------------------------------------------------------------------------


@dataclass
class Agreement:
    n: int
    raw_agreement: float
    cohen_kappa: float
    judge_positive_rate: float
    human_positive_rate: float
    judge_yes_human_no: int
    judge_no_human_yes: int

    def summary(self) -> str:
        return (
            f"n={self.n}  agreement={self.raw_agreement:.1%}  "
            f"kappa={self.cohen_kappa:.2f}  "
            f"(judge lenient: {self.judge_yes_human_no}, "
            f"judge harsh: {self.judge_no_human_yes})"
        )


def cohen_kappa(judge: Sequence[bool], human: Sequence[bool]) -> Agreement:
    """Cohen's kappa for two binary raters.

    Raw agreement alone is misleading on a skewed label set: if 90% of answers
    are grounded, a judge that says "grounded" unconditionally scores 90%
    agreement while carrying zero information. Kappa subtracts the agreement
    expected from the marginals, so that degenerate judge scores 0.

    Interpretation used in the README: <0.4 poor, 0.4-0.6 moderate,
    0.6-0.8 substantial, >0.8 near-ceiling.
    """
    if len(judge) != len(human):
        raise ValueError("judge and human label sequences must be the same length")
    n = len(judge)
    if n == 0:
        return Agreement(0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    both_yes = sum(1 for j, h in zip(judge, human) if j and h)
    both_no = sum(1 for j, h in zip(judge, human) if not j and not h)
    j_yes_h_no = sum(1 for j, h in zip(judge, human) if j and not h)
    j_no_h_yes = sum(1 for j, h in zip(judge, human) if not j and h)

    po = (both_yes + both_no) / n
    pj = (both_yes + j_yes_h_no) / n
    ph = (both_yes + j_no_h_yes) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1 - pe)

    return Agreement(
        n=n,
        raw_agreement=po,
        cohen_kappa=kappa,
        judge_positive_rate=pj,
        human_positive_rate=ph,
        judge_yes_human_no=j_yes_h_no,
        judge_no_human_yes=j_no_h_yes,
    )
