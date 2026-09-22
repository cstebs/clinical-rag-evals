"""Anthropic client wrapper: caching, cost accounting, and batch submission.

Every model call in this project goes through here, for three reasons.

**Caching.** Judge calls dominate the bill. A sweep re-runs the same
(question, answer, context) triples constantly - re-running an unchanged config,
adding one new config, re-checking a single failure. The cache key is a hash of
everything that could change the response (model, system prompt, messages,
schema, effort), so an unchanged call is free and instant, and a changed prompt
correctly misses. Without this, iterating on the harness would be unaffordable.

**Cost accounting.** ``usage`` from every uncached call is accumulated and
priced, so each run record carries what it actually cost. "How do you know it
works?" has a sibling question - "what did it cost to find out?" - and this is
the answer to it.

**Batch.** ``complete_many`` transparently routes to the Message Batches API
above a threshold: same requests, asynchronous, 50% of the price. A full sweep
is exactly the non-latency-sensitive workload batch exists for.

Auth: the SDK resolves credentials itself (``ANTHROPIC_API_KEY``, then
``ANTHROPIC_AUTH_TOKEN``, then an ``ant auth login`` profile). We construct a
zero-argument client and let it do that, rather than reading env vars ourselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths

DEFAULT_MODEL = "claude-opus-5"

# USD per million tokens, Claude Opus 5. Batch runs at 50% of both.
PRICE_INPUT_PER_MTOK = 5.00
PRICE_OUTPUT_PER_MTOK = 25.00
BATCH_DISCOUNT = 0.5

# Below this many pending calls, batch's up-to-an-hour latency is not worth it.
BATCH_THRESHOLD = 20

# Batch polling
BATCH_POLL_SECONDS = 20
BATCH_TIMEOUT_SECONDS = 60 * 60 * 6


@dataclass
class LLMResponse:
    text: str
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse the response as JSON, tolerating a stray code fence.

        With ``output_config.format`` set, the first text block is guaranteed to
        be valid JSON. The fallback path exists for calls made without a schema.
        """
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t
            t = t.rsplit("```", 1)[0]
        return json.loads(t)


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    batched_calls: int = 0

    def cost_usd(self) -> float:
        batch_share = self.batched_calls / self.calls if self.calls else 0.0
        multiplier = 1.0 - batch_share * (1.0 - BATCH_DISCOUNT)
        return multiplier * (
            self.input_tokens / 1e6 * PRICE_INPUT_PER_MTOK
            + self.output_tokens / 1e6 * PRICE_OUTPUT_PER_MTOK
        )

    def summary(self) -> dict[str, Any]:
        return {
            "api_calls": self.calls,
            "cache_hits": self.cached_calls,
            "batched_calls": self.batched_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.cost_usd(), 4),
        }


class LLMUnavailable(RuntimeError):
    """Raised when a model call is required but no credentials are configured."""


@dataclass
class Call:
    """One pending model call."""

    key: str
    system: str
    user: str
    max_tokens: int = 4096
    effort: str = "medium"
    schema: dict[str, Any] | None = None


class LLMClient:
    """Cached, cost-accounted access to the Messages API.

    ``stub`` lets the whole pipeline run with no credentials: it receives a
    :class:`Call` and returns the response text. The test suite uses it, and so
    does ``--stub-llm``, which is how the generation and judging paths can be
    exercised end to end offline. Stubbed responses are never written to the
    cache, so they can never be mistaken for real results later.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        stub: Callable[[Call], str] | None = None,
        use_batch: bool = True,
    ):
        self.model = model
        self.cache_dir = cache_dir or paths.JUDGE_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stub = stub
        self.use_batch = use_batch
        self.usage = Usage()
        self._client = None

    # -- credentials -----------------------------------------------------

    @staticmethod
    def credentials_available() -> bool:
        return bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or (Path.home() / ".config" / "anthropic").exists()
        )

    def _handle(self):
        if self._client is None:
            if not self.credentials_available():
                raise LLMUnavailable(
                    "No Anthropic credentials found. Set ANTHROPIC_API_KEY, or run "
                    "`ant auth login`, or pass --stub-llm to exercise the pipeline "
                    "without model calls."
                )
            import anthropic

            # An identity-linked API key is rejected without the workspace it
            # acts in; the SDK does not read this env var itself.
            workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": workspace} if workspace else None
            self._client = anthropic.Anthropic(default_headers=headers)
        return self._client

    # -- cache -----------------------------------------------------------

    def cache_key(self, call: Call) -> str:
        h = hashlib.sha256()
        for part in (
            self.model,
            call.system,
            call.user,
            str(call.max_tokens),
            call.effort,
            json.dumps(call.schema, sort_keys=True) if call.schema else "",
        ):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key[:2]}" / f"{key}.json"

    def _cache_get(self, key: str) -> LLMResponse | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        payload = json.loads(p.read_text(encoding="utf-8"))
        return LLMResponse(
            text=payload["text"],
            cached=True,
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            stop_reason=payload.get("stop_reason"),
            raw=payload.get("raw", {}),
        )

    def _cache_put(self, key: str, resp: LLMResponse) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "text": resp.text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "stop_reason": resp.stop_reason,
                    "raw": resp.raw,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -- request construction --------------------------------------------

    def _params(self, call: Call) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": call.effort}
        if call.schema:
            output_config["format"] = {"type": "json_schema", "schema": call.schema}
        return {
            "model": self.model,
            "max_tokens": call.max_tokens,
            "system": call.system,
            "messages": [{"role": "user", "content": call.user}],
            # Adaptive thinking on Claude Opus 5. budget_tokens is rejected on
            # this model, and temperature/top_p have been removed entirely.
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
        }

    @staticmethod
    def _text_of(message) -> str:
        return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")

    # -- single call -----------------------------------------------------

    def complete(self, call: Call) -> LLMResponse:
        key = self.cache_key(call)
        hit = self._cache_get(key)
        if hit is not None:
            self.usage.cached_calls += 1
            return hit

        if self.stub is not None:
            # Never cached: a stub result must not contaminate real results.
            return LLMResponse(text=self.stub(call), cached=False)

        client = self._handle()
        params = self._params(call)
        # Streaming keeps a long thinking pass from tripping the HTTP timeout.
        with client.messages.stream(**params) as stream:
            message = stream.get_final_message()

        resp = LLMResponse(
            text=self._text_of(message),
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            stop_reason=message.stop_reason,
        )
        self._record(resp)
        self._cache_put(key, resp)
        return resp

    def _record(self, resp: LLMResponse, batched: bool = False) -> None:
        self.usage.calls += 1
        self.usage.input_tokens += resp.input_tokens
        self.usage.output_tokens += resp.output_tokens
        if batched:
            self.usage.batched_calls += 1

    # -- many calls ------------------------------------------------------

    def complete_many(self, calls: Sequence[Call], progress: bool = True) -> list[LLMResponse]:
        """Resolve a list of calls, using the Batch API when it is worth it.

        Cache hits are served first and never submitted. Only genuine misses
        reach the API, which is why a re-run after a small change costs almost
        nothing.
        """
        results: list[LLMResponse | None] = [None] * len(calls)
        pending: list[int] = []

        for i, call in enumerate(calls):
            hit = self._cache_get(self.cache_key(call))
            if hit is not None:
                self.usage.cached_calls += 1
                results[i] = hit
            else:
                pending.append(i)

        if not pending:
            return [r for r in results]  # type: ignore[misc]

        if self.stub is not None:
            for i in pending:
                results[i] = LLMResponse(text=self.stub(calls[i]), cached=False)
            return [r for r in results]  # type: ignore[misc]

        if self.use_batch and len(pending) >= BATCH_THRESHOLD:
            for i, resp in zip(pending, self._run_batch([calls[i] for i in pending])):
                results[i] = resp
        else:
            from tqdm import tqdm

            it = tqdm(pending, desc="model calls", disable=not progress)
            for i in it:
                results[i] = self.complete(calls[i])

        return [r for r in results]  # type: ignore[misc]

    def _run_batch(self, calls: Sequence[Call]) -> list[LLMResponse]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        client = self._handle()
        # custom_id must be unique within the batch; the cache key already is,
        # but identical calls can appear twice in one sweep, so index it.
        requests = [
            Request(
                custom_id=f"c{i:05d}",
                params=MessageCreateParamsNonStreaming(**self._params(call)),
            )
            for i, call in enumerate(calls)
        ]

        batch = client.messages.batches.create(requests=requests)
        print(f"submitted batch {batch.id} with {len(requests)} requests (50% pricing)")

        deadline = time.monotonic() + BATCH_TIMEOUT_SECONDS
        while True:
            batch = client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"batch {batch.id} did not finish within the timeout")
            print(f"  batch {batch.id}: {batch.processing_status}")
            time.sleep(BATCH_POLL_SECONDS)

        # Results arrive in arbitrary order; key by custom_id, never position.
        by_id: dict[str, LLMResponse] = {}
        for result in client.messages.batches.results(batch.id):
            rtype = result.result.type
            if rtype == "succeeded":
                msg = result.result.message
                resp = LLMResponse(
                    text=self._text_of(msg),
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                    stop_reason=msg.stop_reason,
                )
            else:
                resp = LLMResponse(text="", stop_reason=f"batch_{rtype}")
            by_id[result.custom_id] = resp

        out: list[LLMResponse] = []
        for i, call in enumerate(calls):
            resp = by_id.get(f"c{i:05d}", LLMResponse(text="", stop_reason="batch_missing"))
            self._record(resp, batched=True)
            if resp.text:
                self._cache_put(self.cache_key(call), resp)
            out.append(resp)
        return out


def estimate_cost(
    n_questions: int,
    n_configs: int,
    calls_per_question: int = 2,
    input_tokens_per_call: int = 3500,
    output_tokens_per_call: int = 500,
    batch: bool = True,
) -> dict[str, Any]:
    """Budget a sweep before running it (SPEC section 7.2).

    ``calls_per_question`` defaults to 2: one generation call, one judge call
    that scores groundedness, correctness and abstention together. Scoring the
    three metrics in one call rather than three is a 3x saving on the judge, and
    costs nothing in rigour because they share the same evidence.
    """
    calls = n_questions * n_configs * calls_per_question
    inp = calls * input_tokens_per_call
    out = calls * output_tokens_per_call
    cost = inp / 1e6 * PRICE_INPUT_PER_MTOK + out / 1e6 * PRICE_OUTPUT_PER_MTOK
    if batch:
        cost *= BATCH_DISCOUNT
    return {
        "calls": calls,
        "input_tokens": inp,
        "output_tokens": out,
        "batch_pricing": batch,
        "estimated_cost_usd": round(cost, 2),
        "note": "cache hits are free; a re-run of an unchanged config costs $0",
    }
