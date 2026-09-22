"""Retrieval orchestration: query handling -> index -> optional reranking.

This is the layer the experiment matrix actually varies. It composes three
independent switches, and the discipline rule from SPEC section 4 - change one
variable per run - is enforced structurally: each switch is a separate config
field, and the run record writes all of them down, so a table row can always be
traced back to exactly which knobs were set.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .chunk import Chunk
from .index import CrossEncoderReranker, Hit, build_index
from .llm import Call, LLMClient, LLMUnavailable
from .prompts import Prompt
from .prompts import load as load_prompt

# ---------------------------------------------------------------------------
# query handling
# ---------------------------------------------------------------------------


def rewrite_system_prompt(prompt: Prompt | None = None) -> str:
    """The rewriter's system prompt from ``prompts/rewrite/<version>.md`` (v1 by default)."""
    return (prompt or load_prompt("rewrite", "v1")).render()

REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # minItems/maxItems are rejected by structured outputs; the prompt asks
        # for 2-4 alternates and expand() slices to 5 regardless.
        "queries": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}


class QueryRewriter:
    """LLM query expansion, cached like every other model call.

    Fusion note: each generated query is searched separately and the result
    lists are combined by reciprocal rank fusion, the same mechanism the hybrid
    index uses. Concatenating the queries into one long string instead would
    average the embeddings and usually retrieve less than the original query.
    """

    def __init__(self, llm: LLMClient, rrf_k: int = 60, prompt: Prompt | None = None):
        self.llm = llm
        self.rrf_k = rrf_k
        self.system = rewrite_system_prompt(prompt)

    def expand(self, question: str) -> list[str]:
        call = Call(
            key=f"rewrite::{question}",
            system=self.system,
            user=question,
            max_tokens=1024,
            effort="low",  # a rewrite is not an intelligence-sensitive task
            schema=REWRITE_SCHEMA,
        )
        try:
            queries = self.llm.complete(call).json()["queries"]
        except LLMUnavailable:
            # Never degrade silently. Falling back to the raw query here would
            # produce a run record labelled "+rewrite" in which no rewriting
            # happened - a table row that is quietly a lie.
            raise
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            # A malformed rewrite is recoverable: the original query is still a
            # valid query, and the run record shows the config that was used.
            # Anything else - an auth failure, a connection error - propagates,
            # because falling back on those would produce a row labelled
            # "+rewrite" in which no rewriting happened.
            print(f"  rewrite response malformed ({exc!r}); using the raw query")
            return [question]
        out = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if question not in out:
            out.insert(0, question)
        return out[:5]


# ---------------------------------------------------------------------------
# retriever
# ---------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    index: str = "dense"           # bm25 | dense | hybrid
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_depth: int = 30
    query_rewrite: bool = False
    top_k: int = 10


class Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        config: RetrievalConfig,
        llm: LLMClient | None = None,
        rewrite_prompt: Prompt | None = None,
    ):
        self.chunks = list(chunks)
        self.config = config
        self.index = build_index(self.chunks, config.index, config.embed_model)
        self.reranker = (
            CrossEncoderReranker(config.rerank_model, config.rerank_depth)
            if config.rerank
            else None
        )
        self.rewriter = (
            QueryRewriter(llm, prompt=rewrite_prompt) if config.query_rewrite else None
        )
        if config.query_rewrite and llm is None:
            raise ValueError("query_rewrite=True requires an LLMClient")

    # -- search ----------------------------------------------------------

    def search(self, question: str, k: int | None = None) -> list[Hit]:
        k = k or self.config.top_k

        # Reranking only helps if it is given more candidates than it returns.
        depth = max(k, self.config.rerank_depth) if self.reranker else k

        queries = self.rewriter.expand(question) if self.rewriter else [question]

        if len(queries) == 1:
            scored = self.index.search(queries[0], depth)
        else:
            scored = self._fuse([self.index.search(q, depth) for q in queries], depth)

        hits = [
            Hit(
                chunk_id=self.chunks[i].chunk_id,
                doc_id=self.chunks[i].doc_id,
                score=score,
                rank=rank,
                text=self.chunks[i].text,
                section=self.chunks[i].section,
                page_start=self.chunks[i].page_start,
                page_end=self.chunks[i].page_end,
            )
            for rank, (i, score) in enumerate(scored, start=1)
        ]

        if self.reranker:
            hits = self.reranker.rerank(question, hits, k)
        return hits[:k]

    def _fuse(self, result_lists: Sequence[Sequence[tuple[int, float]]], k: int):
        rrf_k = self.rewriter.rrf_k if self.rewriter else 60
        fused: dict[int, float] = {}
        for hits in result_lists:
            for rank, (idx, _s) in enumerate(hits, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        return sorted(fused.items(), key=lambda kv: -kv[1])[:k]


def format_context(hits: Sequence[Hit], max_chars: int | None = None) -> str:
    """Render hits as numbered, attributed context for the generator.

    Each passage is labelled with a citation tag the generator is told to use.
    Without stable per-passage tags there is no way for either the generator or
    the judge to point at *which* passage supports a claim, and groundedness
    becomes unmeasurable.
    """
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] doc={h.doc_id} pages={h.page_start}-{h.page_end}"
        if h.section:
            header += f" section={h.section!r}"
        body = f"{header}\n{h.text}"
        if max_chars is not None and used + len(body) > max_chars:
            break
        parts.append(body)
        used += len(body)
    return "\n\n---\n\n".join(parts)
