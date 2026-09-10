"""Indexes: BM25 (lexical), dense (embeddings), and hybrid fusion.

Why exact search instead of FAISS or Chroma
-------------------------------------------
An approximate-nearest-neighbour index has its own recall loss. In a harness
whose entire purpose is to attribute recall changes to a chosen variable, an ANN
index adds a second, uncontrolled source of recall loss and confounds every row
in the table. This corpus produces tens of thousands of 384-dimensional vectors;
an exact matrix product over that is a few milliseconds of NumPy. Paying a
correctness tax to save milliseconds would be the wrong trade in a measurement
tool, so search here is exact and the vector store is a NumPy array on disk.

Embedding choice
----------------
``BAAI/bge-small-en-v1.5`` served through ``fastembed`` (ONNX Runtime). Anthropic
has no first-party embeddings endpoint - the Messages API's supporting endpoints
are Batches, Files, Token Counting and Models - so embeddings are either a
third-party API or local. Local wins for a portfolio repo: free, deterministic,
offline after the first model download, and identical weights to the
sentence-transformers version without pulling in a ~2 GB torch dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from . import paths
from .chunk import Chunk

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Corpus embedding is the slowest deterministic step in a cold sweep. ONNX
# Runtime does not saturate a many-core CPU on a model this small - raising
# intra-op threads measured as no help - but fanning batches across worker
# processes does, at roughly 2.4x on a 16-core machine. ``0`` means "one worker
# per core"; a small fixed number avoids oversubscribing when something else is
# running. Only used for the one-off corpus pass, never for single queries,
# where process startup would dominate.
EMBED_WORKERS = 4
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# Reciprocal-rank-fusion constant. 60 is the value from the original Cormack et
# al. paper and the de-facto default; it damps the influence of top ranks just
# enough that one retriever cannot dominate the fusion on its own.
RRF_K = 60


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    text: str
    section: str
    page_start: int
    page_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "section": self.section,
            "pages": [self.page_start, self.page_end],
        }


# ---------------------------------------------------------------------------
# lexical
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Deliberately short. Aggressive stopword removal hurts on regulatory text,
# where phrases like "should not" and "may be" carry real meaning.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those such
    """.split()
)


def lexical_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """Okapi BM25 over the chunk texts."""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([lexical_tokens(c.text) for c in self.chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(lexical_tokens(query))
        return _topk(np.asarray(scores, dtype=np.float32), k)


# ---------------------------------------------------------------------------
# dense
# ---------------------------------------------------------------------------


class DenseIndex:
    """Exact cosine search over normalized ONNX embeddings.

    Embeddings are cached on disk under a key derived from the model id and a
    hash of the chunk texts. A sweep that varies only the retriever therefore
    embeds the corpus once, not once per run - which is the difference between a
    sweep taking minutes and taking an hour.
    """

    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], model_name: str = DEFAULT_EMBED_MODEL):
        self.chunks = list(chunks)
        self.model_name = model_name
        self._model = None
        self.matrix = self._load_or_build()

    # -- embedding -------------------------------------------------------

    def _model_handle(self):
        if self._model is None:
            from fastembed import TextEmbedding

            paths.MODELS.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=self.model_name, cache_dir=str(paths.MODELS)
            )
        return self._model

    def _cache_key(self) -> str:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        for c in self.chunks:
            h.update(c.chunk_id.encode())
            h.update(b"\x00")
            h.update(c.text.encode("utf-8"))
            h.update(b"\x01")
        return h.hexdigest()[:16]

    def _load_or_build(self) -> np.ndarray:
        paths.CACHE.mkdir(parents=True, exist_ok=True)
        cache_file = paths.CACHE / f"emb-{self._cache_key()}.npy"
        if cache_file.exists():
            return np.load(cache_file)

        model = self._model_handle()
        vecs = np.asarray(
            list(
                model.embed(
                    [c.text for c in self.chunks],
                    batch_size=64,
                    parallel=EMBED_WORKERS,
                )
            ),
            dtype=np.float32,
        )
        vecs = _l2_normalize(vecs)
        np.save(cache_file, vecs)
        return vecs

    def embed_query(self, query: str) -> np.ndarray:
        # bge models expect an instruction prefix on the query side only.
        # fastembed's query_embed applies it; falling back to embed() would
        # silently cost a few points of recall.
        model = self._model_handle()
        vec = np.asarray(next(iter(model.query_embed([query]))), dtype=np.float32)
        return _l2_normalize(vec.reshape(1, -1))[0]

    # -- search ----------------------------------------------------------

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embed_query(query)
        scores = self.matrix @ q  # both sides L2-normalized => cosine
        return _topk(scores, k)


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


# ---------------------------------------------------------------------------
# hybrid
# ---------------------------------------------------------------------------


class HybridIndex:
    """Reciprocal rank fusion of BM25 and dense results.

    RRF fuses *ranks*, not scores, which is the point: BM25 scores are unbounded
    and corpus-dependent while cosine scores sit in [-1, 1]. Any weighted sum of
    the two requires a normalization scheme that is itself a hidden tuned
    parameter. RRF needs no normalization and no per-corpus calibration.
    """

    name = "hybrid"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        model_name: str = DEFAULT_EMBED_MODEL,
        rrf_k: int = RRF_K,
        candidate_depth: int = 100,
    ):
        self.chunks = list(chunks)
        self.bm25 = BM25Index(chunks)
        self.dense = DenseIndex(chunks, model_name=model_name)
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        depth = max(self.candidate_depth, k)
        fused: dict[int, float] = {}
        for hits in (self.bm25.search(query, depth), self.dense.search(query, depth)):
            for rank, (idx, _score) in enumerate(hits, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(idx, score) for idx, score in ordered]


# ---------------------------------------------------------------------------
# reranking
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Cross-encoder reranking over a deeper candidate pool.

    A bi-encoder scores query and passage independently, so it can never notice
    that a passage mentions the query's exact entity. A cross-encoder reads both
    together and is far more accurate - and far too slow to run over the whole
    corpus. Hence: retrieve ``candidate_depth`` cheaply, rerank those, keep k.
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, candidate_depth: int = 30):
        self.model_name = model_name
        self.candidate_depth = candidate_depth
        self._model = None

    def _handle(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            paths.MODELS.mkdir(parents=True, exist_ok=True)
            self._model = TextCrossEncoder(
                model_name=self.model_name, cache_dir=str(paths.MODELS)
            )
        return self._model

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = list(
            self._handle().rerank(query, [h.text for h in hits], batch_size=32)
        )
        order = np.argsort(-np.asarray(scores, dtype=np.float32))[:k]
        out: list[Hit] = []
        for new_rank, i in enumerate(order, start=1):
            h = hits[int(i)]
            out.append(
                Hit(
                    chunk_id=h.chunk_id,
                    doc_id=h.doc_id,
                    score=float(scores[int(i)]),
                    rank=new_rank,
                    text=h.text,
                    section=h.section,
                    page_start=h.page_start,
                    page_end=h.page_end,
                )
            )
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _topk(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = min(k, scores.shape[0])
    if k <= 0:
        return []
    # argpartition is O(n); a full sort of 40k scores per query is wasteful.
    part = np.argpartition(-scores, k - 1)[:k]
    part = part[np.argsort(-scores[part])]
    return [(int(i), float(scores[int(i)])) for i in part]


def build_index(
    chunks: Sequence[Chunk],
    kind: str = "dense",
    model_name: str = DEFAULT_EMBED_MODEL,
):
    if kind == "bm25":
        return BM25Index(chunks)
    if kind == "dense":
        return DenseIndex(chunks, model_name=model_name)
    if kind == "hybrid":
        return HybridIndex(chunks, model_name=model_name)
    raise ValueError(f"unknown index kind: {kind!r}")
