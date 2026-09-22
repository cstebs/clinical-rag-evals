"""Chunking strategies.

Three strategies, deliberately ordered from structure-blind to structure-aware,
because the delta between them is one of the rows in the results table:

``fixed``
    A sliding window of N tokens over the document's flat text. Ignores
    paragraphs, headings, sentences - everything. This is the weak baseline the
    project starts from on purpose (SPEC section 4): a baseline that is already
    good leaves no visible headroom.

``recursive``
    Split on the largest natural boundary that fits (blank line, then sentence,
    then whitespace), packing pieces up to the target size. This is the
    industry-default splitter and the honest mid-point.

``section``
    Detect headings from the PDF's own typography (font size and weight,
    recorded during ingest) and never let a chunk cross a heading. Each chunk is
    prefixed with its heading path, so "Section IV.B. Informed Consent" travels
    with the text into the embedding.

Every strategy emits the same :class:`Chunk` record, so the index layer does not
know or care which one produced its input.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from . import paths
from .tokenize import Tokenizer, get_tokenizer, truncation_report


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    n_tokens: int
    page_start: int
    page_end: int
    section: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# document loading
# ---------------------------------------------------------------------------


@dataclass
class Line:
    page: int
    text: str
    size: float
    bold: bool


@dataclass
class Document:
    doc_id: str
    title: str
    lines: list[Line]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def load_document(doc_id: str) -> Document:
    payload = json.loads((paths.PROCESSED / f"{doc_id}.json").read_text(encoding="utf-8"))
    return Document(
        doc_id=payload["doc_id"],
        title=payload["title"],
        lines=[Line(line["page"], line["t"], line["size"], line["bold"]) for line in payload["lines"]],
    )


# ---------------------------------------------------------------------------
# heading detection
# ---------------------------------------------------------------------------

# FDA guidance uses a fairly consistent outline style. These catch the common
# forms even when the typography is flat.
#
# Every numeric form requires a period after the number ("3. Excipients") or a
# multi-level marker ("3.2 Study Design"). An earlier version accepted a bare
# leading integer, which made street addresses ("10903 New Hampshire Ave."),
# footnote markers ("4 In this guidance, the term...") and dates ("27 April
# 1995") all read as headings. Every form also requires the following word to
# start with a capital, which rejects mid-sentence line wraps like "VII. for
# additional discussion."
_HEADING_PATTERNS = [
    re.compile(r"^(?:[IVXLC]+\.)\s+[A-Z]"),                  # "IV. DISCUSSION"
    re.compile(r"^(?:[A-Z]\.)\s+[A-Z]"),                     # "B. Informed Consent"
    re.compile(r"^\d+\.\s+[A-Z]"),                          # "3. Excipients"
    re.compile(r"^\d+(?:\.\d+)+\.?\s+[A-Z]"),               # "3.2 Study Design"
    re.compile(r"^(?:APPENDIX|GLOSSARY|REFERENCES|ABBREVIATIONS)\b", re.I),
]

# A leading outline marker, stripped before judging capitalisation.
_MARKER = re.compile(r"^(?:[IVXLC]+\.|[A-Z]\.|\d+(?:\.\d+)*\.?)\s+")

# Four or more dots in a row is a table-of-contents leader. TOC lines are
# typographically identical to the headings they point at, so without this the
# front matter fires a section break for every entry and then the real heading
# fires it again.
_DOT_LEADER = re.compile(r"\.{4,}")

# Words that stay lowercase in title case and so carry no signal either way.
_MINOR_WORDS = frozenset(
    "a an and as at but by for from in into nor of on or the to with via per vs".split()
)

# A heading is a complete noun phrase. A line ending in one of these is either
# the first half of a heading that wrapped onto a second line, or an entry in a
# numbered reference list - both of which used to fire a spurious section break.
# Rejecting them also means a wrapped heading breaks once, on its first line,
# with the remainder falling into the section body where it belongs.
_TRAILING_FUNCTION_WORDS = frozenset(
    "a an and as at be been by for from in is are into of on or that the their "
    "these this to which with".split()
)

_MAX_HEADING_TOKENS = 18

# Inside a reference list, "14. Task Force on Risk Management. Report to the FDA
# Commissioner..." is an entry, not a subsection - and it is indistinguishable
# from one by pattern, font or capitalisation. The only thing that separates
# them is where in the document they sit, so this is handled with one bit of
# state in _section rather than in _is_heading.
_REFERENCE_SECTION = re.compile(r"REFERENC|BIBLIOGRAPH", re.I)
_NUMBERED = re.compile(r"^\d")


def _title_like(text: str) -> bool:
    """Is this capitalised like a heading rather than like a sentence?

    Headings are Title Case or ALL CAPS; body text is sentence case. This is
    what separates the numbered subsection "1. Statistical Considerations" from
    the numbered list item "1. Clinical trials should be conducted in accordance
    with the ethical principles that", which no pattern or font size can tell
    apart.
    """
    words = [w for w in _MARKER.sub("", text).split() if any(c.isalpha() for c in w)]
    words = [w for w in words if w.lower().strip(".,()") not in _MINOR_WORDS]
    if not words:
        return False
    capitalised = sum(1 for w in words if w.lstrip("(“\"'")[:1].isupper())
    # One stray lowercase word (an abbreviation, a hyphenated tail) is tolerated;
    # a sentence will fail this by a wide margin.
    return capitalised >= len(words) - 1


def _is_heading(line: Line, body_size: float) -> bool:
    t = line.text.strip()
    if not t or len(t) > 140:
        return False
    words = t.split()
    if len(words) > _MAX_HEADING_TOKENS:
        return False

    # A table-of-contents entry points at a heading; it is not one.
    if _DOT_LEADER.search(t):
        return False
    # Anything set smaller than the body is fine print: footnotes, the address
    # block on the cover page, figure captions. Never a section heading.
    if line.size > 0 and line.size < body_size - 0.1:
        return False
    # Headings do not end like clauses. Applied before the pattern check, not
    # after: previously an outline marker exempted a line from this test, which
    # is exactly how numbered list items ("3. Evidence of positive dechallenge
    # or positive rechallenge;") got in.
    if t.endswith((",", ";", ":")):
        return False
    if t.endswith(".") and not t.isupper():
        return False
    if words[-1].lower().strip(".,;:)”\"'") in _TRAILING_FUNCTION_WORDS:
        return False

    typographic = line.size >= body_size + 0.6 or (line.bold and line.size >= body_size - 0.1)
    structural = any(p.match(t) for p in _HEADING_PATTERNS)
    all_caps = t.isupper() and len(words) >= 2

    # Typography alone is noisy (headers/footers are often bold too), so require
    # either a strong typographic signal plus brevity, or an outline marker.
    if not (structural or (typographic and all_caps)):
        return False
    return _title_like(t)


def _body_size(lines: Sequence[Line]) -> float:
    sizes = [line.size for line in lines if line.size > 0]
    if not sizes:
        return 10.0
    try:
        return statistics.mode(round(s, 1) for s in sizes)
    except statistics.StatisticsError:
        return statistics.median(sizes)


# Running headers/footers repeat on nearly every page and add nothing but noise
# to the index. Drop any short line that appears on more than 60% of pages.
def _boilerplate(lines: Sequence[Line], n_pages: int) -> set[str]:
    if n_pages < 4:
        return set()
    seen: dict[str, set[int]] = {}
    for line in lines:
        key = line.text.strip()
        if 0 < len(key) <= 90:
            seen.setdefault(key, set()).add(line.page)
    cutoff = 0.6 * n_pages
    return {k for k, pages in seen.items() if len(pages) > cutoff}


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------


def chunk_document(
    doc: Document,
    strategy: str = "recursive",
    target_tokens: int = 512,
    overlap_tokens: int = 0,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    tok = tokenizer or get_tokenizer()
    n_pages = max((line.page for line in doc.lines), default=1)
    drop = _boilerplate(doc.lines, n_pages)
    lines = [line for line in doc.lines if line.text.strip() not in drop]
    if not lines:
        return []

    if strategy == "fixed":
        pieces = _fixed(lines, tok, target_tokens, overlap_tokens)
    elif strategy == "recursive":
        pieces = _recursive(lines, tok, target_tokens, overlap_tokens)
    elif strategy == "section":
        pieces = _section(lines, tok, target_tokens, overlap_tokens)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy!r}")

    out: list[Chunk] = []
    for i, (text, p0, p1, section) in enumerate(pieces):
        text = text.strip()
        if not text:
            continue
        out.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{i:04d}",
                doc_id=doc.doc_id,
                ordinal=i,
                text=text,
                n_tokens=tok.count(text),
                page_start=p0,
                page_end=p1,
                section=section,
            )
        )
    return out


_Piece = tuple[str, int, int, str]


def _fixed(lines: Sequence[Line], tok: Tokenizer, size: int, overlap: int) -> list[_Piece]:
    """Sliding token window over the flat text. Structure-blind by design."""
    # Track which page each token came from so a chunk can still cite pages.
    tokens: list[str] = []
    token_pages: list[int] = []
    for line in lines:
        pieces = tok.encode_pieces(line.text + " ")
        tokens.extend(pieces)
        token_pages.extend([line.page] * len(pieces))

    if not tokens:
        return []

    step = max(1, size - overlap)
    out: list[_Piece] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        pages = token_pages[start : start + size]
        out.append((_detokenize(window), pages[0], pages[-1], ""))
        if start + size >= len(tokens):
            break
    return out


_SUBWORD = re.compile(r"^##")


def _detokenize(pieces: Sequence[str]) -> str:
    """Reassemble WordPiece tokens into readable text.

    The fixed strategy is the only one that operates below the line level, so
    it is the only one that needs this. The output is not byte-identical to the
    source - that is inherent to slicing mid-word, and is part of why the fixed
    baseline underperforms.
    """
    out: list[str] = []
    for p in pieces:
        if _SUBWORD.match(p):
            out.append(p[2:])
        elif out and p in ".,;:)]}%'\"" or p in ("n't",):
            out.append(p)
        else:
            out.append(" " + p if out else p)
    return "".join(out).strip()


_PARA_BREAK = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _recursive(lines: Sequence[Line], tok: Tokenizer, size: int, overlap: int) -> list[_Piece]:
    """Pack text into chunks, splitting on the largest boundary that fits."""
    units = _line_units(lines)
    return _pack(units, tok, size, overlap, section="")


def _section(lines: Sequence[Line], tok: Tokenizer, size: int, overlap: int) -> list[_Piece]:
    """Split at headings first, then pack within each section."""
    body = _body_size(lines)
    out: list[_Piece] = []
    current: list[tuple[str, int]] = []
    heading = ""
    heading_page = 1

    def flush() -> None:
        nonlocal current
        if current:
            out.extend(_pack(current, tok, size, overlap, section=heading))
        elif heading:
            # A heading with nothing under it - two headings in a row, or a
            # section that is only a title. _pack is what normally carries the
            # heading text into a chunk, so without this branch the line would
            # be dropped from the corpus entirely.
            out.append((heading, heading_page, heading_page, heading))
        current = []

    in_references = False
    for line in lines:
        candidate = _is_heading(line, body)
        if candidate and in_references and _NUMBERED.match(line.text.strip()):
            candidate = False  # a numbered entry in a reference list
        if candidate:
            flush()
            heading = line.text.strip()
            heading_page = line.page
            in_references = bool(_REFERENCE_SECTION.search(heading))
            # The heading is deliberately *not* appended to the body. _pack
            # prefixes it to every chunk in the section; keeping it here too
            # made the first chunk of every section repeat its own heading,
            # which polluted both the embedding and the BM25 term counts.
        else:
            current.append((line.text, line.page))
    flush()
    return out


def _line_units(lines: Sequence[Line]) -> list[tuple[str, int]]:
    return [(line.text, line.page) for line in lines]


def _pack(
    units: Sequence[tuple[str, int]],
    tok: Tokenizer,
    size: int,
    overlap: int,
    section: str,
) -> list[_Piece]:
    """Greedy packing with sentence-level fallback for oversized units."""
    # The heading prefix is prepended to every chunk in this section, so it has
    # to come out of the budget. Without this, a "512-token" section chunk is
    # actually 512 plus the heading, which pushes it past the encoder's 512-token
    # window and truncates the tail - penalising the section strategy for an
    # accounting mistake rather than for anything real about the strategy.
    prefix = f"{section}\n\n" if section else ""
    size = max(32, size - tok.count(prefix)) if prefix else size

    out: list[_Piece] = []
    buf: list[tuple[str, int]] = []
    buf_tokens = 0
    # How many leading units of ``buf`` were carried over from the previous
    # chunk as overlap. Without this, a document whose last unit exactly fills a
    # chunk would emit a final chunk containing nothing but the carried tail -
    # a pure duplicate that inflates the chunk count and the index.
    carried = 0

    def flush() -> None:
        nonlocal buf, buf_tokens, carried
        if not buf:
            return
        text = " ".join(t for t, _ in buf)
        out.append((prefix + text, buf[0][1], buf[-1][1], section))
        if overlap > 0:
            tail: list[tuple[str, int]] = []
            acc = 0
            for unit in reversed(buf):
                n = tok.count(unit[0])
                if acc + n > overlap:
                    break
                tail.insert(0, unit)
                acc += n
            buf, buf_tokens, carried = tail, acc, len(tail)
        else:
            buf, buf_tokens, carried = [], 0, 0

    def append(unit: tuple[str, int], n: int) -> None:
        nonlocal buf_tokens
        buf.append(unit)
        buf_tokens += n

    for text, page in units:
        text = text.strip()
        if not text:
            continue
        n = tok.count(text)

        if n > size:
            # A single line longer than the target: split it into sentences.
            flush()
            for sent in _SENTENCE.split(text):
                sent = sent.strip()
                if not sent:
                    continue
                sn = tok.count(sent)
                if buf_tokens + sn > size and buf:
                    flush()
                append((sent, page), sn)
            continue

        if buf_tokens + n > size and buf:
            flush()
        append((text, page), n)

    # Emit the remainder only if it holds text that is not already in the
    # previous chunk, and without re-seeding the overlap buffer.
    if len(buf) > carried:
        text = " ".join(t for t, _ in buf)
        out.append((prefix + text, buf[0][1], buf[-1][1], section))
    return out


# ---------------------------------------------------------------------------
# corpus-level entry point
# ---------------------------------------------------------------------------


def chunk_corpus(
    doc_ids: Iterable[str],
    strategy: str = "recursive",
    target_tokens: int = 512,
    overlap_tokens: int = 0,
) -> tuple[list[Chunk], dict[str, Any]]:
    """Chunk every document and return the chunks plus a diagnostics block.

    The diagnostics land in the run record so that a row in the results table
    can be read alongside how much text the encoder actually saw.
    """
    tok = get_tokenizer()
    chunks: list[Chunk] = []
    for doc_id in doc_ids:
        doc = load_document(doc_id)
        chunks.extend(
            chunk_document(doc, strategy, target_tokens, overlap_tokens, tokenizer=tok)
        )

    counts = [c.n_tokens for c in chunks]
    stats: dict[str, Any] = {
        "strategy": strategy,
        "target_tokens": target_tokens,
        "overlap_tokens": overlap_tokens,
        "n_chunks": len(chunks),
        "mean_tokens": round(statistics.mean(counts), 1) if counts else 0.0,
        "median_tokens": statistics.median(counts) if counts else 0,
        "p95_tokens": (
            sorted(counts)[int(0.95 * (len(counts) - 1))] if counts else 0
        ),
        "sections_detected": sum(1 for c in chunks if c.section),
        **tok.describe(),
        **truncation_report(counts),
    }
    return chunks, stats
