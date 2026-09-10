"""Token counting, using the embedding model's own tokenizer.

Why this matters more than it looks: ``bge-small-en-v1.5`` has a 512-token
maximum sequence length. Anything longer is silently truncated by the encoder -
no warning, no error, just a chunk whose tail never reaches the vector. A
harness that measures chunk size in *characters* or in "roughly 4 chars per
token" will report a 1024-token configuration as if all 1024 tokens were
embedded, and the resulting table will be quietly wrong.

So chunk sizes here are counted with the same tokenizer the encoder uses, and
:func:`truncation_report` makes the loss visible in the run record instead of
hiding it.
"""

from __future__ import annotations

import functools
import re

# Max sequence length of BAAI/bge-small-en-v1.5 (and of most BERT-family
# encoders). Chunks longer than this lose their tail at embedding time.
MAX_SEQUENCE_TOKENS = 512

_WORD = re.compile(r"\w+|[^\w\s]")


class Tokenizer:
    """Wraps a HuggingFace tokenizer, with a deterministic offline fallback.

    The fallback (regex word/punctuation split) is *not* equivalent to the real
    tokenizer, so it is reported in the run record. It exists so that the test
    suite and a fully offline clone still run end to end.
    """

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5"):
        self.model_id = model_id
        self.exact = False
        self._tok = None
        try:
            from tokenizers import Tokenizer as HFTokenizer

            self._tok = HFTokenizer.from_pretrained(model_id)
            self._tok.no_truncation()
            self._tok.no_padding()
            self.exact = True
        except Exception:
            self._tok = None

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        # Fallback: word+punct pieces, scaled toward subword granularity.
        return max(1, int(len(_WORD.findall(text)) * 1.15))

    def encode_pieces(self, text: str) -> list[str]:
        """Token strings, used only by the fixed-size chunker."""
        if self._tok is not None:
            return self._tok.encode(text, add_special_tokens=False).tokens
        return _WORD.findall(text)

    def describe(self) -> dict[str, object]:
        return {
            "tokenizer": self.model_id if self.exact else "regex-fallback",
            "exact": self.exact,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        }


@functools.lru_cache(maxsize=4)
def get_tokenizer(model_id: str = "BAAI/bge-small-en-v1.5") -> Tokenizer:
    return Tokenizer(model_id)


def truncation_report(token_counts: list[int]) -> dict[str, float]:
    """How much of the corpus the encoder will silently drop.

    ``tokens_lost_pct`` is the share of all chunk tokens that fall beyond the
    encoder's 512-token window and therefore never influence the embedding.
    """
    if not token_counts:
        return {"chunks_over_limit": 0, "chunks_over_limit_pct": 0.0, "tokens_lost_pct": 0.0}
    total = sum(token_counts)
    lost = sum(max(0, n - MAX_SEQUENCE_TOKENS) for n in token_counts)
    over = sum(1 for n in token_counts if n > MAX_SEQUENCE_TOKENS)
    return {
        "chunks_over_limit": over,
        "chunks_over_limit_pct": round(100.0 * over / len(token_counts), 2),
        "tokens_lost_pct": round(100.0 * lost / max(total, 1), 2),
    }
