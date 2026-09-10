"""The gold question set: schema, loading, and integrity validation.

Question types (SPEC section 3.1)
--------------------------------
``fact``
    One passage, one unambiguous answer. Baseline sanity - if these fail,
    nothing downstream is worth reading.

``multi_hop``
    The answer requires combining two or more documents, so ``source_docs`` has
    more than one entry and partial retrieval scores partial recall.

``near_miss``
    A plausible-but-wrong passage scores highly on similarity. These separate
    retrievers that understand the query from retrievers that match its
    vocabulary, and they are where reranking earns or fails to earn its cost.

``negative``
    The corpus genuinely cannot answer it. ``source_docs`` is empty. The system
    must decline. Most public RAG demos contain zero of these, which is exactly
    why they are the most informative rows in the table: a confabulated answer
    to an unanswerable question is the failure that matters in a regulated
    domain, and nothing else in the eval detects it.

Integrity
---------
:func:`validate` re-reads the parsed corpus and checks that every
``source_quote`` actually occurs in the document it is attributed to. This
catches three separate classes of bug at once: a mislabelled document id, a
quote that was paraphrased rather than copied, and a document whose PDF parse
silently produced garbage. Any of the three would corrupt the eval while looking
completely healthy from the outside.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import paths

QUESTION_TYPES = ("fact", "multi_hop", "near_miss", "negative")


@dataclass
class GoldQuestion:
    qid: str
    question: str
    question_type: str
    expected_answer: str
    source_docs: list[str] = field(default_factory=list)
    source_quotes: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_negative(self) -> bool:
        return self.question_type == "negative"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load(path: Path | None = None) -> list[GoldQuestion]:
    path = path or paths.GOLD
    out: list[GoldQuestion] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                out.append(GoldQuestion(**json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return out


def save(questions: Iterable[GoldQuestion], path: Path | None = None) -> None:
    path = path or paths.GOLD
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for q in questions:
            fh.write(q.to_json() + "\n")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Collapse the differences PDF extraction introduces but meaning does not.

    Ligatures, curly quotes, en-dashes and line-wrap whitespace all differ
    between what a human copies out of a PDF viewer and what PyMuPDF emits.
    Comparing raw strings would produce false failures on correct labels.
    """
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _WS.sub(" ", s).strip().lower()


@dataclass
class ValidationIssue:
    qid: str
    severity: str  # "error" | "warning"
    message: str


def validate(
    questions: list[GoldQuestion],
    known_doc_ids: set[str] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    doc_text_cache: dict[str, str] = {}

    def doc_text(doc_id: str) -> str | None:
        if doc_id not in doc_text_cache:
            p = paths.PROCESSED / f"{doc_id}.json"
            if not p.exists():
                return None
            payload = json.loads(p.read_text(encoding="utf-8"))
            doc_text_cache[doc_id] = _normalize(
                " ".join(l["t"] for l in payload["lines"])
            )
        return doc_text_cache[doc_id]

    for q in questions:
        if q.qid in seen_ids:
            issues.append(ValidationIssue(q.qid, "error", "duplicate qid"))
        seen_ids.add(q.qid)

        if q.question_type not in QUESTION_TYPES:
            issues.append(
                ValidationIssue(
                    q.qid, "error", f"unknown question_type {q.question_type!r}"
                )
            )

        if q.is_negative:
            if q.source_docs:
                issues.append(
                    ValidationIssue(
                        q.qid, "error", "negative question must have no source_docs"
                    )
                )
            continue

        if not q.source_docs:
            issues.append(
                ValidationIssue(q.qid, "error", "answerable question has no source_docs")
            )
        if q.question_type == "multi_hop" and len(q.source_docs) < 2:
            issues.append(
                ValidationIssue(
                    q.qid, "error", "multi_hop question cites fewer than 2 documents"
                )
            )

        for doc_id in q.source_docs:
            if known_doc_ids is not None and doc_id not in known_doc_ids:
                issues.append(
                    ValidationIssue(q.qid, "error", f"{doc_id} is not in the manifest")
                )

        # Every quote must appear verbatim in one of the cited documents.
        for quote in q.source_quotes:
            needle = _normalize(quote)
            if len(needle) < 15:
                issues.append(
                    ValidationIssue(q.qid, "warning", f"quote too short to verify: {quote!r}")
                )
                continue
            found = False
            missing_corpus = False
            for doc_id in q.source_docs:
                hay = doc_text(doc_id)
                if hay is None:
                    missing_corpus = True
                    continue
                if needle in hay:
                    found = True
                    break
            if missing_corpus and not found:
                issues.append(
                    ValidationIssue(
                        q.qid,
                        "warning",
                        "cited document not parsed locally; run `rag-evals ingest` first",
                    )
                )
            elif not found:
                issues.append(
                    ValidationIssue(
                        q.qid,
                        "error",
                        f"quote not found in cited documents: {quote[:70]!r}",
                    )
                )

    return issues


def summarize(questions: list[GoldQuestion]) -> dict[str, Any]:
    counts: dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    for q in questions:
        counts[q.question_type] = counts.get(q.question_type, 0) + 1
    return {
        "n_questions": len(questions),
        "by_type": counts,
        "n_documents_cited": len({d for q in questions for d in q.source_docs}),
    }
