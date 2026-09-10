"""Answer synthesis over retrieved context.

The generator is deliberately plain. It is not the thing being measured, and
every clever trick added here (self-consistency, multi-step reasoning over the
passages, re-retrieval) would become a confound in every row of the table. It
does exactly two things that matter to the eval:

1. **Cites passages by tag.** Claims carry ``[1]``-style references back to the
   numbered context blocks. Without this, groundedness cannot be checked at the
   claim level - only vibed at the answer level.

2. **Abstains explicitly.** When the context does not contain the answer, the
   model must emit the exact abstention marker. A structured field, not a phrase
   match, is what makes abstention accuracy a real measurement rather than a
   regex guessing at hedging language like "I'm not certain, but...".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .index import Hit
from .llm import Call, LLMClient
from .retrieve import format_context

ABSTAIN_MARKER = "INSUFFICIENT_CONTEXT"

GENERATE_SYSTEM = f"""You answer questions about FDA guidance for industry using \
only the numbered passages provided.

Rules:
- Use only the passages. Do not use prior knowledge about FDA guidance, even if \
you are confident it is correct.
- Every factual claim must cite the passage it came from, as [1], [2], and so on. \
A sentence may cite more than one passage.
- If the passages do not contain enough information to answer, set \
"abstained" to true and set "answer" to exactly "{ABSTAIN_MARKER}". Do not \
partially answer and do not guess. Declining is the correct behaviour when the \
evidence is absent, and is scored as correct.
- Do not hedge as a substitute for abstaining. Either the passages support an \
answer, or they do not.
- Be direct and concise. Regulatory language matters: preserve "should" versus \
"must" exactly as the passage states it."""

GENERATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "abstained": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited_passages": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["abstained", "answer", "cited_passages"],
    "additionalProperties": False,
}


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    cited_passages: list[int]
    context: str
    hit_doc_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.text,
            "abstained": self.abstained,
            "cited_passages": self.cited_passages,
        }


def build_call(question: str, hits: Sequence[Hit], max_context_chars: int = 24000) -> Call:
    context = format_context(hits, max_chars=max_context_chars)
    user = f"Passages:\n\n{context}\n\n---\n\nQuestion: {question}"
    return Call(
        key=f"generate::{question}",
        system=GENERATE_SYSTEM,
        user=user,
        max_tokens=2048,
        effort="medium",
        schema=GENERATE_SCHEMA,
    )


def parse_answer(question: str, hits: Sequence[Hit], raw_text: str, context: str) -> Answer:
    try:
        import json

        data = json.loads(raw_text)
    except Exception:
        # A malformed response is a real failure and must not be silently
        # laundered into an abstention - that would inflate abstention accuracy.
        return Answer(
            question=question,
            text=raw_text.strip(),
            abstained=False,
            cited_passages=[],
            context=context,
            hit_doc_ids=[h.doc_id for h in hits],
        )

    answer = str(data.get("answer", "")).strip()
    abstained = bool(data.get("abstained")) or answer == ABSTAIN_MARKER
    cited = [int(c) for c in data.get("cited_passages", []) if isinstance(c, (int, float))]
    return Answer(
        question=question,
        text=answer,
        abstained=abstained,
        cited_passages=cited,
        context=context,
        hit_doc_ids=[h.doc_id for h in hits],
    )


def generate_batch(
    llm: LLMClient,
    questions: Sequence[str],
    hit_lists: Sequence[Sequence[Hit]],
    max_context_chars: int = 24000,
) -> list[Answer]:
    calls = [build_call(q, h, max_context_chars) for q, h in zip(questions, hit_lists)]
    responses = llm.complete_many(calls)
    return [
        parse_answer(q, h, r.text, c.user)
        for q, h, r, c in zip(questions, hit_lists, responses, calls)
    ]
