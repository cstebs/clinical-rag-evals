"""The LLM judge, and the machinery for validating it.

An unvalidated LLM judge is vibes with extra steps. The number that makes this
project worth reading is not any retrieval metric - it is the agreement between
this judge and hand labels on a held-out subset, published with the
disagreements shown. That is what ``rag-evals validate-judge`` produces.

Design decisions that matter
----------------------------

**One call scores all three metrics.** Groundedness, correctness and abstention
all depend on the same evidence: the question, the retrieved context, the
expected answer, and the produced answer. Splitting them into three calls
triples the bill and gives three inconsistent readings of the same evidence.

**The judge sees the expected answer for correctness, and must ignore it for
groundedness.** These are genuinely different questions. An answer can be
correct in substance while citing nothing that supports it - that is a
confabulation that happened to land right, and it is the single most dangerous
failure mode in a regulated domain. The prompt states the separation explicitly
and the schema keeps the two verdicts apart.

**Reasoning is stored, not just the score.** ``judge_reasoning`` is written into
the run record for every item. The failure-analysis section of the README is
only writable because you can read *why* the judge ruled the way it did - and
the judge-validation step is only actionable because you can see where it and a
human diverged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .generate import ABSTAIN_MARKER, Answer
from .gold import GoldQuestion
from .llm import Call, LLMClient
from .prompts import Prompt
from .prompts import load as load_prompt


def system_prompt(prompt: Prompt | None = None) -> str:
    """The judge's system prompt, rendered from ``prompts/judge/<version>.md``.

    The judge prompt is the measurement instrument, which is exactly why it is
    versioned and hashed into every run record rather than sitting inline where
    an edit would silently change what "grounded" means for every later row.
    With no argument this returns v1, the text every published verdict and the
    40 hand-label agreements were produced against.
    """
    return (prompt or load_prompt("judge", "v1")).render(abstain_marker=ABSTAIN_MARKER)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "correct": {"type": "boolean"},
        "abstained": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["grounded", "correct", "abstained", "unsupported_claims", "reasoning"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    qid: str
    grounded: bool
    correct: bool
    abstained: bool
    unsupported_claims: list[str]
    reasoning: str
    parse_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_call(
    q: GoldQuestion,
    answer: Answer,
    max_context_chars: int = 24000,
    system: str | None = None,
) -> Call:
    reference = (
        q.expected_answer
        if not q.is_negative
        else "(No reference answer: this question is NOT answerable from the corpus. "
        "The correct behaviour is to abstain.)"
    )
    user = (
        f"QUESTION:\n{q.question}\n\n"
        f"RETRIEVED PASSAGES:\n{answer.context[:max_context_chars]}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"SYSTEM ANSWER:\n{answer.text or '(empty)'}"
    )
    return Call(
        key=f"judge::{q.qid}",
        system=system if system is not None else system_prompt(),
        user=user,
        max_tokens=2048,
        # Grading is the intelligence-sensitive step in this pipeline: it is the
        # measurement instrument, and a sloppy instrument invalidates the table.
        effort="high",
        schema=JUDGE_SCHEMA,
    )


def parse_verdict(qid: str, raw_text: str, fallback_abstained: bool) -> Verdict:
    try:
        data = json.loads(raw_text)
    except Exception:
        return Verdict(
            qid=qid,
            grounded=False,
            correct=False,
            abstained=fallback_abstained,
            unsupported_claims=[],
            reasoning=f"judge response could not be parsed: {raw_text[:200]!r}",
            parse_failed=True,
        )
    return Verdict(
        qid=qid,
        grounded=bool(data.get("grounded")),
        correct=bool(data.get("correct")),
        abstained=bool(data.get("abstained")),
        unsupported_claims=[str(c) for c in data.get("unsupported_claims", [])],
        reasoning=str(data.get("reasoning", "")),
    )


def judge_batch(
    llm: LLMClient,
    questions: Sequence[GoldQuestion],
    answers: Sequence[Answer],
    max_context_chars: int = 24000,
    prompt: Prompt | None = None,
) -> list[Verdict]:
    system = system_prompt(prompt)
    calls = [build_call(q, a, max_context_chars, system) for q, a in zip(questions, answers)]
    responses = llm.complete_many(calls)
    return [
        parse_verdict(q.qid, r.text, a.abstained)
        for q, a, r in zip(questions, answers, responses)
    ]


# ---------------------------------------------------------------------------
# judge validation
# ---------------------------------------------------------------------------

HUMAN_LABEL_FIELDS = ("grounded", "correct", "abstained")


@dataclass
class HumanLabel:
    """One hand-graded item from ``evals/human_labels.jsonl``.

    ``run_id`` pins the label to the exact run whose answers were graded. A
    label is only meaningful against the answer text it was written for, so
    re-labelling is required when the answers change - silently reusing old
    labels against new answers would produce an agreement number that means
    nothing.
    """

    qid: str
    run_id: str
    grounded: bool
    correct: bool
    abstained: bool
    note: str = ""


class HumanLabelError(ValueError):
    """A line of ``human_labels.jsonl`` could not be read.

    Carries the file and 1-based line number so a hand-edit typo (the usual
    case is a dropped closing quote on ``note``) is reported as a location to
    fix rather than a JSON traceback.
    """

    def __init__(self, path, lineno: int, reason: str, text: str) -> None:
        self.path, self.lineno, self.reason, self.text = path, lineno, reason, text
        snippet = text if len(text) <= 120 else text[:117] + "..."
        super().__init__(f"{path}:{lineno}: {reason}\n    {snippet}")


_HUMAN_LABEL_KEYS = {"qid", "run_id", *HUMAN_LABEL_FIELDS, "note"}


def load_human_labels(path) -> list[HumanLabel]:
    """Read every label, or raise :class:`HumanLabelError` naming the bad line.

    The file is hand-edited, so it is linted as it is read: malformed JSON,
    a non-object line, missing or unknown keys, and non-boolean verdicts are
    each reported with the line number. Lines starting with ``//`` are comments.
    """
    out: list[HumanLabel] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise HumanLabelError(path, lineno, f"invalid JSON: {e.msg}, column {e.colno}", line) from None
            if not isinstance(rec, dict):
                raise HumanLabelError(path, lineno, "expected a JSON object", line)
            missing = sorted(_HUMAN_LABEL_KEYS - {"note"} - rec.keys())
            unknown = sorted(rec.keys() - _HUMAN_LABEL_KEYS)
            if missing:
                raise HumanLabelError(path, lineno, f"missing keys {missing}", line)
            if unknown:
                raise HumanLabelError(path, lineno, f"unknown keys {unknown}", line)
            for f in HUMAN_LABEL_FIELDS:
                if not isinstance(rec[f], bool):
                    raise HumanLabelError(path, lineno, f"{f!r} must be true or false, got {rec[f]!r}", line)
            out.append(HumanLabel(**rec))
    return out


def append_human_label(path, label: HumanLabel) -> None:
    """Append one label immediately.

    Written per item rather than batched at the end so that an interrupted
    labelling session keeps everything already decided. Labelling is the
    slowest human step in the project; losing forty minutes of it to a stray
    Ctrl-C would be an unforced error.
    """
    from dataclasses import asdict

    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")


def select_for_labelling(
    items: Sequence[tuple[str, str]],
    already: set[str],
    n: int,
    seed: int = 11,
) -> list[str]:
    """Choose a stratified sample of qids to hand-label.

    Two deliberate biases:

    * **Every negative case is taken first.** They are the questions where a
      judge is most likely to be wrong (declining correctly is a subtler call
      than matching an answer), and they are only 12 of 85, so a proportional
      sample would include four of them and measure almost nothing about the
      behaviour that matters most.
    * **The remainder is proportional** to what is left, so the sample still
      reflects the shape of the gold set rather than becoming all negatives.

    Deterministic given ``seed``, and it skips anything already labelled, so a
    resumed session continues rather than re-drawing a different sample.
    """
    import random

    pool = [(qid, qtype) for qid, qtype in items if qid not in already]
    if n <= 0 or not pool:
        return []

    rng = random.Random(seed)
    by_type: dict[str, list[str]] = {}
    for qid, qtype in pool:
        by_type.setdefault(qtype, []).append(qid)
    for qids in by_type.values():
        rng.shuffle(qids)

    chosen: list[str] = []
    negatives = by_type.pop("negative", [])
    chosen.extend(negatives[:n])

    remaining = n - len(chosen)
    if remaining > 0 and by_type:
        total = sum(len(v) for v in by_type.values())
        # Largest-remainder allocation, so rounding never overshoots n.
        quotas = {t: remaining * len(v) / total for t, v in by_type.items()}
        alloc = {t: int(q) for t, q in quotas.items()}
        for t in sorted(by_type, key=lambda t: -(quotas[t] - alloc[t])):
            if sum(alloc.values()) >= remaining:
                break
            alloc[t] += 1
        for t, k in alloc.items():
            chosen.extend(by_type[t][:k])

    # Stable presentation order, independent of the draw.
    return sorted(chosen)
