"""Chunker invariants.

The properties tested here are the ones that, if violated, would corrupt every
row of the results table without producing an error: chunks that drop text,
chunks that silently ignore the size target, and section chunks that leak across
a heading boundary.
"""

import pytest

from rag_evals.chunk import Chunk, Document, Line, chunk_document
from rag_evals.tokenize import Tokenizer, truncation_report


def _doc(lines):
    return Document(doc_id="test-1", title="Test", lines=lines)


def _body(text, page=1, size=10.0):
    return Line(page=page, text=text, size=size, bold=False)


def _heading(text, page=1, size=14.0):
    return Line(page=page, text=text, size=size, bold=True)


SENTENCES = [
    "The sponsor should submit the protocol to the reviewing IRB before enrolment begins.",
    "Investigators must retain records for two years after the marketing application is approved.",
    "FDA recommends that data monitoring committees operate under a written charter.",
    "Informed consent documents should describe reasonably foreseeable risks to the subject.",
]


class TestStrategies:
    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "section"])
    def test_produces_chunks_and_ids_are_unique(self, strategy):
        doc = _doc([_body(s) for s in SENTENCES * 20])
        chunks = chunk_document(doc, strategy=strategy, target_tokens=128)
        assert chunks
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all(c.doc_id == "test-1" for c in chunks)

    @pytest.mark.parametrize("strategy", ["recursive", "section"])
    def test_no_text_is_dropped(self, strategy):
        """Every source sentence must survive into some chunk.

        A chunker that quietly loses the tail of a document produces a corpus
        smaller than the one in the manifest, and every recall number computed
        on it is wrong in a direction nothing else would reveal.
        """
        doc = _doc([_body(s) for s in SENTENCES * 10])
        chunks = chunk_document(doc, strategy=strategy, target_tokens=64)
        joined = " ".join(c.text for c in chunks)
        for s in SENTENCES:
            assert s in joined

    def test_recursive_respects_target_size(self):
        doc = _doc([_body(s) for s in SENTENCES * 30])
        tok = Tokenizer()
        chunks = chunk_document(doc, strategy="recursive", target_tokens=100)
        # A chunk may exceed the target only when a single indivisible unit
        # does; with these short sentences none do.
        assert all(c.n_tokens <= 130 for c in chunks), max(c.n_tokens for c in chunks)

    def test_pages_are_tracked(self):
        doc = _doc([_body(s, page=i + 1) for i, s in enumerate(SENTENCES)])
        chunks = chunk_document(doc, strategy="recursive", target_tokens=1000)
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 4


class TestSectionAware:
    def test_chunks_do_not_cross_headings(self):
        doc = _doc(
            [_heading("IV. INFORMED CONSENT")]
            + [_body(SENTENCES[0])] * 3
            + [_heading("V. RECORD RETENTION")]
            + [_body(SENTENCES[1])] * 3
        )
        chunks = chunk_document(doc, strategy="section", target_tokens=4000)
        # Large target: without heading awareness this would be a single chunk.
        assert len(chunks) >= 2
        for c in chunks:
            assert not ("INFORMED CONSENT" in c.text and "RECORD RETENTION" in c.text)

    def test_heading_path_travels_with_the_text(self):
        doc = _doc([_heading("IV. INFORMED CONSENT")] + [_body(SENTENCES[0])] * 3)
        chunks = chunk_document(doc, strategy="section", target_tokens=1000)
        assert chunks[0].section
        # The heading is prefixed so it reaches the embedding, not just metadata.
        assert "INFORMED CONSENT" in chunks[0].text

    def test_outline_markers_detected_without_typography(self):
        """Many FDA PDFs have flat typography; outline markers must still work."""
        doc = _doc(
            [_body("A. Scope")]
            + [_body(SENTENCES[0])] * 2
            + [_body("B. Definitions")]
            + [_body(SENTENCES[1])] * 2
        )
        chunks = chunk_document(doc, strategy="section", target_tokens=4000)
        assert len(chunks) >= 2


class TestOverlap:
    def test_overlap_repeats_tail(self):
        doc = _doc([_body(s) for s in SENTENCES * 8])
        no_ov = chunk_document(doc, strategy="recursive", target_tokens=60, overlap_tokens=0)
        with_ov = chunk_document(doc, strategy="recursive", target_tokens=60, overlap_tokens=30)
        # Overlap duplicates text, so it always yields at least as many chunks.
        assert len(with_ov) >= len(no_ov)
        total_no = sum(c.n_tokens for c in no_ov)
        total_ov = sum(c.n_tokens for c in with_ov)
        assert total_ov > total_no


class TestBoilerplate:
    def test_repeated_headers_are_stripped(self):
        """A footer on every page is pure index noise."""
        lines = []
        for page in range(1, 11):
            lines.append(_body("Contains Nonbinding Recommendations", page=page))
            lines.append(_body(SENTENCES[page % len(SENTENCES)], page=page))
        doc = _doc(lines)
        chunks = chunk_document(doc, strategy="recursive", target_tokens=500)
        joined = " ".join(c.text for c in chunks)
        assert "Contains Nonbinding Recommendations" not in joined

    def test_short_documents_keep_everything(self):
        """The boilerplate heuristic must not fire on a 2-page document."""
        lines = [_body("Contains Nonbinding Recommendations", page=p) for p in (1, 2)]
        lines += [_body(SENTENCES[0], page=1)]
        doc = _doc(lines)
        chunks = chunk_document(doc, strategy="recursive", target_tokens=500)
        assert "Contains Nonbinding Recommendations" in " ".join(c.text for c in chunks)


class TestTruncationReport:
    def test_reports_encoder_loss(self):
        # 512 is the encoder limit; a 1024-token chunk loses half its tokens.
        rep = truncation_report([1024, 512])
        assert rep["chunks_over_limit"] == 1
        assert rep["tokens_lost_pct"] == pytest.approx(100 * 512 / 1536, abs=0.1)

    def test_empty(self):
        assert truncation_report([])["tokens_lost_pct"] == 0.0


class TestOverlapTail:
    def test_no_duplicate_trailing_chunk(self):
        """With overlap on, the final chunk must contain new text.

        A document whose last unit exactly fills a chunk would otherwise leave
        only the carried-over tail in the buffer and emit it as a final chunk -
        a pure duplicate that inflates the chunk count and pollutes the index
        with a passage that adds no information.
        """
        doc = _doc([_body(s) for s in SENTENCES * 12])
        chunks = chunk_document(doc, strategy="recursive", target_tokens=48, overlap_tokens=24)
        assert len(chunks) >= 2
        last, prev = chunks[-1].text, chunks[-2].text
        assert last not in prev, "final chunk is wholly contained in the previous one"


class TestHeadingPrecision:
    """The heading detector is the whole of the section strategy.

    An earlier version matched a bare leading integer, so street addresses,
    footnote markers and numbered list items all counted as headings. That
    inflated the "sections detected" figure with junk and, worse, made the
    published finding about section-aware chunking a statement about a broken
    detector rather than about the strategy.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "5600 Fishers Lane, Rockville, MD 20857 (Tel) 301-827-4573",
            "10903 New Hampshire Ave., Bldg. 71, Room 3128",
            "3. Evidence of positive dechallenge or positive rechallenge;",
            "27 April 1995",
            "4 In this guidance, the term investigational product refers to drugs",
            "VII. for additional discussion.",
            "1. Clinical trials should be conducted in accordance with the ethical",
            "INTRODUCTION AND SCOPE ....................................... 1",
            "21 CFR 610.9, prior to product licensing, the alternative method",
        ],
    )
    def test_junk_is_not_a_heading(self, text):
        doc = _doc([_body(text)] + [_body(SENTENCES[0])] * 2)
        chunks = chunk_document(doc, strategy="section", target_tokens=4000)
        assert all(not c.section for c in chunks), chunks[0].section

    @pytest.mark.parametrize(
        "text",
        [
            "IV. DISCUSSION",
            "B. Informed Consent",
            "3. Excipients",
            "3.2 Study Design",
            "5.1.14 RELREC Domain",
            "APPENDIX B",
        ],
    )
    def test_real_headings_survive(self, text):
        """Outline markers work on their own, with no typographic signal."""
        doc = _doc([_body(text)] + [_body(SENTENCES[0])] * 2)
        chunks = chunk_document(doc, strategy="section", target_tokens=4000)
        assert chunks[0].section == text

    def test_all_caps_heading_needs_a_typographic_signal(self):
        """An unmarked all-caps line is a heading only if it is set apart.

        Bold or larger than the body, it is a heading. At body weight and size
        it is just as likely to be an acronym-heavy sentence fragment lifted out
        of a table, so it is left in the body.
        """
        text = "TRIAL DESIGN CONSIDERATIONS"
        marked = _doc([_heading(text)] + [_body(SENTENCES[0])] * 2)
        assert chunk_document(marked, strategy="section", target_tokens=4000)[0].section == text
        flat = _doc([_body(text)] + [_body(SENTENCES[0])] * 2)
        assert all(not c.section for c in chunk_document(flat, strategy="section", target_tokens=4000))

    def test_footnote_sized_text_is_never_a_heading(self):
        """Footnotes are set below the body size and often start with a number."""
        lines = [_body(SENTENCES[0], size=12.0)] * 3
        lines.append(Line(page=1, text="8 The expected sample size is the average.", size=10.0, bold=False))
        lines += [_body(SENTENCES[1], size=12.0)] * 3
        chunks = chunk_document(_doc(lines), strategy="section", target_tokens=4000)
        assert all(not c.section for c in chunks)

    def test_numbered_reference_entries_do_not_split_the_reference_list(self):
        lines = [_heading("REFERENCES")]
        lines += [
            _body("12. Fox MP, MacLehose RF, and TL Lash, 2021, Applying Bias Analysis"),
            _body("13. ICH Guidance on Viral Safety Evaluation of Biotechnology Products"),
        ]
        chunks = chunk_document(_doc(lines), strategy="section", target_tokens=4000)
        assert {c.section for c in chunks} == {"REFERENCES"}


class TestHeadingNotDuplicated:
    def test_heading_appears_once_in_its_first_chunk(self):
        """The prefix carries the heading; the body must not repeat it.

        _pack prepends the heading to every chunk in a section. _section used to
        also leave the heading line in the section body, so the first chunk of
        every section began with its heading twice - polluting the embedding and
        double-counting every heading term in BM25.
        """
        doc = _doc([_heading("IV. INFORMED CONSENT")] + [_body(SENTENCES[0])] * 3)
        chunks = chunk_document(doc, strategy="section", target_tokens=1000)
        assert chunks[0].text.count("IV. INFORMED CONSENT") == 1

    def test_heading_with_no_body_is_still_kept(self):
        """Two headings in a row must not silently drop the first."""
        doc = _doc(
            [_heading("IV. INFORMED CONSENT"), _heading("V. RECORD RETENTION")]
            + [_body(SENTENCES[0])] * 2
        )
        chunks = chunk_document(doc, strategy="section", target_tokens=1000)
        joined = " ".join(c.text for c in chunks)
        assert "IV. INFORMED CONSENT" in joined
        assert "V. RECORD RETENTION" in joined
