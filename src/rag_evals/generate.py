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
from .prompts import Prompt, load as load_prompt
from .retrieve import format_context

ABSTAIN_MARKER = "INSUFFICIENT_CONTEXT"


def system_prompt(prompt: Prompt | None = None) -> str:
    """The generator's system prompt, rendered from ``prompts/generate/<version>.md``.

    The text is not inline here on purpose: it is versioned, hashed into every
    run record, and immutable once a result references it. See
    :mod:`rag_evals.prompts`. With no argument this returns the v1 text, which
    is what every pre-versioning result ran on.
    """
    return (prompt or load_prompt("generate", "v1")).render(abstain_marker=ABSTAIN_MARKER)

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


def build_call(
    question: str,
    hits: Sequence[Hit],
    max_context_chars: int = 24000,
    system: str | None = None,
) -> Call:
    context = format_context(hits, max_chars=max_context_chars)
    user = f"Passages:\n\n{context}\n\n---\n\nQuestion: {question}"
    return Call(
        key=f"generate::{question}",
        system=system if system is not None else system_prompt(),
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
    prompt: Prompt | None = None,
) -> list[Answer]:
    system = system_prompt(prompt)
    calls = [build_call(q, h, max_context_chars, system) for q, h in zip(questions, hit_lists)]
    responses = llm.complete_many(calls)
    return [
        parse_answer(q, h, r.text, c.user)
        for q, h, r, c in zip(questions, hit_lists, responses, calls)
    ]
