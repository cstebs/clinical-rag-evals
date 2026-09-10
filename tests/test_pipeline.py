"""Gold-set validation, config discipline, LLM caching, and index behaviour."""

import json

import numpy as np
import pytest

from rag_evals import gold as gold_mod
from rag_evals import paths
from rag_evals.chunk import Chunk
from rag_evals.config import RunConfig, isolation_warning, ChunkingConfig
from rag_evals.index import BM25Index, HybridIndex, _topk, lexical_tokens
from rag_evals.llm import Call, LLMClient, LLMResponse, estimate_cost
from rag_evals.retrieve import RetrievalConfig
from rag_evals.generate import parse_answer, ABSTAIN_MARKER
from rag_evals.judge import parse_verdict


# ---------------------------------------------------------------------------
# gold set
# ---------------------------------------------------------------------------


def _q(**kw):
    base = dict(
        qid="q1",
        question="Q?",
        question_type="fact",
        expected_answer="A.",
        source_docs=["d1"],
        source_quotes=[],
    )
    base.update(kw)
    return gold_mod.GoldQuestion(**base)


class TestGoldValidation:
    def test_negative_with_sources_is_an_error(self):
        issues = gold_mod.validate([_q(question_type="negative", source_docs=["d1"])])
        assert any(i.severity == "error" and "no source_docs" in i.message for i in issues)

    def test_multi_hop_needs_two_documents(self):
        issues = gold_mod.validate([_q(question_type="multi_hop", source_docs=["d1"])])
        assert any("fewer than 2" in i.message for i in issues)

    def test_answerable_needs_a_source(self):
        issues = gold_mod.validate([_q(source_docs=[])])
        assert any("no source_docs" in i.message for i in issues)

    def test_duplicate_qid(self):
        issues = gold_mod.validate([_q(), _q()])
        assert any("duplicate qid" in i.message for i in issues)

    def test_unknown_document(self):
        issues = gold_mod.validate([_q(source_docs=["nope"])], known_doc_ids={"d1"})
        assert any("not in the manifest" in i.message for i in issues)

    def test_quote_verification_normalizes_pdf_artifacts(self, tmp_path, monkeypatch):
        """A quote with curly quotes and a ligature must still match the parse."""
        monkeypatch.setattr(paths, "PROCESSED", tmp_path)
        (tmp_path / "d1.json").write_text(
            json.dumps(
                {
                    "doc_id": "d1",
                    "lines": [{"t": "The sponsor's notification must be filed."}],
                }
            ),
            encoding="utf-8",
        )
        ok = gold_mod.validate([_q(source_quotes=["The sponsor’s notification must be filed"])])
        assert not [i for i in ok if i.severity == "error"]

        bad = gold_mod.validate([_q(source_quotes=["a quote that is simply not present"])])
        assert any("quote not found" in i.message for i in bad)

    def test_summarize_counts_types(self):
        s = gold_mod.summarize([_q(), _q(qid="q2", question_type="negative", source_docs=[])])
        assert s["n_questions"] == 2
        assert s["by_type"]["negative"] == 1


# ---------------------------------------------------------------------------
# config discipline
# ---------------------------------------------------------------------------


class TestConfigDiscipline:
    def test_single_variable_passes(self):
        base = RunConfig(run_id="a")
        cfg = RunConfig(run_id="b", retrieval=RetrievalConfig(index="hybrid"))
        assert isolation_warning(cfg, base) is None

    def test_two_variables_warn(self):
        base = RunConfig(run_id="a")
        cfg = RunConfig(
            run_id="b",
            chunking=ChunkingConfig(strategy="section"),
            retrieval=RetrievalConfig(index="hybrid"),
        )
        warn = isolation_warning(cfg, base)
        assert warn and "2 variables" in warn

    def test_rerank_and_its_depth_count_as_one(self):
        base = RunConfig(run_id="a")
        cfg = RunConfig(
            run_id="b", retrieval=RetrievalConfig(rerank=True, rerank_depth=50)
        )
        assert isolation_warning(cfg, base) is None

    def test_shipped_configs_are_each_a_single_delta(self):
        """The repo's own matrix must obey its own rule."""
        cfgs = {p.stem: RunConfig.load(p) for p in sorted(paths.CONFIGS.glob("*.yaml"))}
        assert cfgs, "no configs shipped"
        for cfg in cfgs.values():
            if not cfg.baseline:
                continue
            assert cfg.baseline in cfgs, f"{cfg.run_id} names a missing baseline"
            assert isolation_warning(cfg, cfgs[cfg.baseline]) is None, cfg.run_id


# ---------------------------------------------------------------------------
# llm caching
# ---------------------------------------------------------------------------


class TestLLMCache:
    def test_key_changes_with_every_input(self, tmp_path):
        c = LLMClient(cache_dir=tmp_path)
        base = Call(key="k", system="S", user="U")
        k0 = c.cache_key(base)
        assert c.cache_key(Call(key="k", system="S2", user="U")) != k0
        assert c.cache_key(Call(key="k", system="S", user="U2")) != k0
        assert c.cache_key(Call(key="k", system="S", user="U", effort="high")) != k0
        assert c.cache_key(Call(key="k", system="S", user="U", schema={"a": 1})) != k0
        # The human-readable key field is not part of the hash.
        assert c.cache_key(Call(key="other", system="S", user="U")) == k0

    def test_model_is_part_of_the_key(self, tmp_path):
        a = LLMClient(model="claude-opus-5", cache_dir=tmp_path)
        b = LLMClient(model="claude-sonnet-5", cache_dir=tmp_path)
        call = Call(key="k", system="S", user="U")
        assert a.cache_key(call) != b.cache_key(call)

    def test_roundtrip_is_a_hit(self, tmp_path):
        c = LLMClient(cache_dir=tmp_path)
        call = Call(key="k", system="S", user="U")
        c._cache_put(c.cache_key(call), LLMResponse(text="hello", input_tokens=5, output_tokens=2))
        got = c.complete(call)
        assert got.text == "hello" and got.cached
        assert c.usage.cached_calls == 1
        assert c.usage.calls == 0  # a cache hit is not an API call

    def test_stub_results_are_never_cached(self, tmp_path):
        """A stubbed answer must not be mistakable for a real one later."""
        c = LLMClient(cache_dir=tmp_path, stub=lambda call: "stubbed")
        call = Call(key="k", system="S", user="U")
        assert c.complete(call).text == "stubbed"
        assert not list(tmp_path.rglob("*.json"))

    def test_cost_estimate_halves_under_batch(self):
        sync = estimate_cost(80, 12, batch=False)["estimated_cost_usd"]
        batched = estimate_cost(80, 12, batch=True)["estimated_cost_usd"]
        assert batched == pytest.approx(sync / 2, rel=1e-3)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_abstention_marker_recognised(self):
        a = parse_answer(
            "q", [], json.dumps({"abstained": False, "answer": ABSTAIN_MARKER, "cited_passages": []}), ""
        )
        assert a.abstained

    def test_malformed_generation_is_not_laundered_into_an_abstention(self):
        """Silently treating a broken response as an abstention would inflate
        abstention accuracy, which is exactly the metric under test."""
        a = parse_answer("q", [], "not json at all", "")
        assert a.abstained is False
        assert a.text == "not json at all"

    def test_malformed_judge_is_flagged(self):
        v = parse_verdict("q1", "<<garbage>>", fallback_abstained=True)
        assert v.parse_failed and not v.grounded


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def _chunks(texts):
    return [
        Chunk(
            chunk_id=f"d{i}::0000",
            doc_id=f"d{i}",
            ordinal=0,
            text=t,
            n_tokens=len(t.split()),
            page_start=1,
            page_end=1,
            section="",
        )
        for i, t in enumerate(texts)
    ]


class TestIndexing:
    def test_topk_is_sorted_descending(self):
        scores = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
        got = _topk(scores, 3)
        assert [i for i, _ in got] == [1, 3, 2]

    def test_topk_handles_k_larger_than_corpus(self):
        assert len(_topk(np.array([1.0, 2.0], dtype=np.float32), 10)) == 2

    def test_bm25_finds_the_exact_term(self):
        idx = BM25Index(
            _chunks(
                [
                    "Informed consent must describe reasonably foreseeable risks.",
                    "Stability testing of the drug substance follows ICH Q1A.",
                    "The data monitoring committee operates under a written charter.",
                ]
            )
        )
        top = idx.search("data monitoring committee charter", 1)
        assert top[0][0] == 2

    def test_lexical_tokens_keep_modal_verbs(self):
        """'should' vs 'must' is meaning in regulatory text, not a stopword."""
        toks = lexical_tokens("The sponsor should notify FDA but must not delay.")
        assert "should" in toks and "must" in toks and "not" in toks
        assert "the" not in toks


# ---------------------------------------------------------------------------
# judge validation sampling
# ---------------------------------------------------------------------------


class TestLabelSampling:
    ITEMS = (
        [(f"q{i:03d}", "fact") for i in range(1, 46)]
        + [(f"q{i:03d}", "multi_hop") for i in range(46, 60)]
        + [(f"q{i:03d}", "near_miss") for i in range(60, 74)]
        + [(f"q{i:03d}", "negative") for i in range(74, 86)]
    )

    def test_takes_every_negative_first(self):
        """Negatives are 12 of 85; a proportional sample would include ~6.

        They are the questions where the judge is most likely to be wrong, so
        the sampler deliberately over-weights them.
        """
        from rag_evals.judge import select_for_labelling

        types = dict(self.ITEMS)
        sel = select_for_labelling(self.ITEMS, set(), 40)
        assert len(sel) == 40
        assert sum(1 for q in sel if types[q] == "negative") == 12

    def test_remainder_is_proportional(self):
        from rag_evals.judge import select_for_labelling

        types = dict(self.ITEMS)
        sel = select_for_labelling(self.ITEMS, set(), 40)
        # 28 non-negative slots split across 45/14/14 -> roughly 17/5/5
        assert sum(1 for q in sel if types[q] == "fact") >= 15

    def test_deterministic(self):
        from rag_evals.judge import select_for_labelling

        assert select_for_labelling(self.ITEMS, set(), 30) == select_for_labelling(
            self.ITEMS, set(), 30
        )

    def test_resume_skips_labelled_and_never_redraws_them(self):
        from rag_evals.judge import select_for_labelling

        first = select_for_labelling(self.ITEMS, set(), 20)
        second = select_for_labelling(self.ITEMS, set(first), 20)
        assert not set(first) & set(second)

    def test_never_exceeds_n(self):
        from rag_evals.judge import select_for_labelling

        for n in (1, 7, 12, 13, 40, 85):
            assert len(select_for_labelling(self.ITEMS, set(), n)) == n

    def test_caps_at_pool_size(self):
        from rag_evals.judge import select_for_labelling

        assert len(select_for_labelling(self.ITEMS, set(), 500)) == len(self.ITEMS)


class TestLabelWriting:
    def test_append_is_incremental_and_readable(self, tmp_path):
        """Each label lands on disk immediately, so a Ctrl-C loses nothing."""
        from rag_evals.judge import HumanLabel, append_human_label, load_human_labels

        path = tmp_path / "human_labels.jsonl"
        append_human_label(path, HumanLabel("q001", "009-rerank", True, True, False, "ok"))
        append_human_label(path, HumanLabel("q002", "009-rerank", False, False, True, ""))
        back = load_human_labels(path)
        assert [l.qid for l in back] == ["q001", "q002"]
        assert back[0].note == "ok"
        assert back[1].abstained is True

    def test_malformed_line_is_reported_by_number(self, tmp_path):
        """A hand-edit typo names the line to fix instead of raising a JSON traceback."""
        from rag_evals.judge import HumanLabelError, load_human_labels

        path = tmp_path / "human_labels.jsonl"
        good = '{"qid":"q001","run_id":"r","grounded":true,"correct":true,"abstained":false,"note":""}'
        path.write_text(
            "// comment header\n"
            + good + "\n"
            + '{"qid":"q002","run_id":"r","grounded":true,"correct":true,"abstained":false,"note":"dropped quote}\n',
            encoding="utf-8",
        )
        with pytest.raises(HumanLabelError) as ei:
            load_human_labels(path)
        assert ei.value.lineno == 3
        assert "invalid JSON" in str(ei.value)
        assert str(path) in str(ei.value)

    @pytest.mark.parametrize(
        "line, expect",
        [
            ('{"qid":"q1","run_id":"r","grounded":true,"correct":true,"note":""}', "missing keys ['abstained']"),
            ('{"qid":"q1","run_id":"r","grounded":true,"correct":true,"abstained":false,"note":"","extra":1}', "unknown keys ['extra']"),
            ('{"qid":"q1","run_id":"r","grounded":"yes","correct":true,"abstained":false,"note":""}', "'grounded' must be true or false"),
            ('[1, 2, 3]', "expected a JSON object"),
        ],
    )
    def test_schema_problems_are_reported_by_number(self, tmp_path, line, expect):
        from rag_evals.judge import HumanLabelError, load_human_labels

        path = tmp_path / "human_labels.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(HumanLabelError) as ei:
            load_human_labels(path)
        assert ei.value.lineno == 1
        assert expect in str(ei.value)
