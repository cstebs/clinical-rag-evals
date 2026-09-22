"""Corpus acquisition: select -> fetch -> parse -> manifest.

Source
------
FDA publishes the full guidance-document index as a public JSON file that backs
the DataTable on the "Search for FDA Guidance Documents" page:

    https://www.fda.gov/files/api/datatables/static/search-for-guidance.json

Each row carries the title, issuing center, communication type, issue date,
docket number, and an anchor tag pointing at the PDF under ``/media/<id>/download``.
We select a subset from that index rather than crawling the site.

Politeness
----------
``https://www.fda.gov/robots.txt`` declares ``Crawl-Delay: 30`` for ``User-agent: *``.
This module reads that file at runtime with :mod:`urllib.robotparser`, refuses to
fetch any URL the file disallows, and sleeps for the *declared* delay between
requests rather than a hardcoded guess. ``--delay`` can only be used to slow the
crawl down, never to speed it past what robots.txt asks for.

What is committed
-----------------
``data/manifest.json`` only: URLs, SHA-256 checksums, byte sizes, fetch dates,
and parse statistics. The PDFs and extracted text stay gitignored, so a clone
reproduces the exact corpus by re-running ``rag-evals ingest`` and comparing
checksums, without redistributing the documents themselves.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.robotparser
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from . import paths

INDEX_URL = "https://www.fda.gov/files/api/datatables/static/search-for-guidance.json"
BASE = "https://www.fda.gov"
ROBOTS_URL = f"{BASE}/robots.txt"

USER_AGENT = (
    "rag-evals/0.1 (portfolio retrieval-eval research corpus; "
    "+https://github.com/cstebs/rag-evals)"
)

# Fallback only. The real value is read from robots.txt at runtime.
FALLBACK_CRAWL_DELAY = 30.0

_TAG = re.compile(r"<[^>]+>")
_HREF = re.compile(r'href="([^"]+)"')
_MEDIA_ID = re.compile(r"/media/(\d+)/download")

# ---------------------------------------------------------------------------
# Corpus scope
#
# Narrow and deep beats broad and shallow (SPEC section 7.3). We take final
# guidance documents from the two drug/biologic centers whose subject matter is
# clinical research, which is the domain the gold questions are authored in.
# Broadening to all 2,788 rows would make the gold set unauthorable by hand.
# ---------------------------------------------------------------------------

DEFAULT_CENTERS = (
    "Center for Drug Evaluation and Research",
    "Center for Biologics Evaluation and Research",
)

DEFAULT_TOPIC_PATTERN = (
    r"clinical trial|clinical stud|clinical investigat|informed consent|"
    r"good clinical practice|\bGCP\b|institutional review|\bIRB\b|"
    r"investigational new drug|\bIND\b|protocol|adaptive design|endpoint|"
    r"patient-reported|data monitoring|electronic (source|records)|"
    r"bioavailability|bioequivalence|safety reporting|pharmacovigilance|"
    r"expanded access|decentralized|real-world|master protocol|enrichment|"
    r"multiregional|statistic"
)


@dataclass
class DocRecord:
    """One corpus document, as recorded in ``data/manifest.json``."""

    doc_id: str
    title: str
    url: str
    landing_page: str
    center: str
    communication_type: str
    issue_date: str
    docket: str
    topics: str

    # populated by fetch()
    sha256: str | None = None
    bytes: int | None = None
    fetched_at: str | None = None

    # populated by parse()
    n_pages: int | None = None
    n_chars: int | None = None
    chars_per_page: float | None = None
    parse_ok: bool | None = None
    parse_note: str = ""


@dataclass
class Manifest:
    created_at: str
    index_url: str
    selection: dict[str, Any]
    docs: list[DocRecord] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            "created_at": self.created_at,
            "index_url": self.index_url,
            "selection": self.selection,
            "n_docs": len(self.docs),
            "docs": [asdict(d) for d in self.docs],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path | None = None) -> Manifest:
        path = path or paths.MANIFEST
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            created_at=raw["created_at"],
            index_url=raw["index_url"],
            selection=raw["selection"],
            docs=[DocRecord(**d) for d in raw["docs"]],
        )

    def save(self, path: Path | None = None) -> None:
        path = path or paths.MANIFEST
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


class Robots:
    """Live robots.txt gate. Fails closed on a parse error."""

    def __init__(self, url: str = ROBOTS_URL, user_agent: str = USER_AGENT):
        self.user_agent = user_agent
        self._rp = urllib.robotparser.RobotFileParser()
        self._rp.set_url(url)
        resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
        resp.raise_for_status()
        self._rp.parse(resp.text.splitlines())
        self.raw = resp.text

    def allowed(self, url: str) -> bool:
        return self._rp.can_fetch(self.user_agent, url)

    def crawl_delay(self) -> float:
        # RobotFileParser matches the wildcard group for an unknown agent.
        declared = self._rp.crawl_delay(self.user_agent)
        if declared is None:
            return FALLBACK_CRAWL_DELAY
        return float(declared)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def _strip_tags(s: str) -> str:
    return html.unescape(_TAG.sub("", s or "")).strip()


def _first_href(s: str) -> str | None:
    m = _HREF.search(s or "")
    return m.group(1) if m else None


def fetch_index(session: requests.Session) -> list[dict[str, Any]]:
    resp = session.get(INDEX_URL, timeout=120)
    resp.raise_for_status()
    return resp.json()


def select_docs(
    rows: Iterable[dict[str, Any]],
    centers: Iterable[str] = DEFAULT_CENTERS,
    topic_pattern: str = DEFAULT_TOPIC_PATTERN,
    communication_type: str = "Guidance Document",
    final_only: bool = True,
    limit: int | None = None,
) -> list[DocRecord]:
    """Filter the FDA index down to the working corpus.

    Selection is deterministic: after filtering, rows are sorted by document id
    so that two runs of ``ingest`` on the same index produce the same corpus in
    the same order. ``limit`` then truncates from a stable position rather than
    from whatever order the upstream JSON happened to arrive in.
    """
    centers = set(centers)
    topic_re = re.compile(topic_pattern, re.I)
    out: list[DocRecord] = []
    seen: set[str] = set()

    for row in rows:
        if row.get("field_center") not in centers:
            continue
        if communication_type and row.get("field_communication_type") != communication_type:
            continue
        if final_only and row.get("field_final_guidance_1") != "Final":
            continue

        media = row.get("field_associated_media_2") or ""
        href = _first_href(media)
        if not href:
            continue
        m = _MEDIA_ID.search(href)
        if not m:
            continue

        title = _strip_tags(row.get("title", ""))
        topics = _strip_tags(row.get("topics-product", ""))
        haystack = f"{title} {topics} {_strip_tags(row.get('field_health_topics',''))}"
        if topic_re and not topic_re.search(haystack):
            continue

        doc_id = f"fda-{m.group(1)}"
        if doc_id in seen:
            continue
        seen.add(doc_id)

        out.append(
            DocRecord(
                doc_id=doc_id,
                title=title,
                url=f"{BASE}/media/{m.group(1)}/download",
                landing_page=BASE + (_first_href(row.get("title", "")) or ""),
                center=row.get("field_center", ""),
                communication_type=row.get("field_communication_type", ""),
                issue_date=row.get("field_issue_datetime", ""),
                docket=_strip_tags(row.get("field_docket_number", "")),
                topics=topics,
            )
        )

    out.sort(key=lambda d: int(d.doc_id.split("-")[1]))
    if limit is not None:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def fetch_docs(
    docs: list[DocRecord],
    robots: Robots,
    delay: float | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> list[DocRecord]:
    """Download each PDF, honouring robots.txt.

    Already-present files are skipped and their checksum recomputed, so an
    interrupted crawl resumes rather than restarting.
    """
    session = session or _session()
    declared = robots.crawl_delay()
    # --delay may only slow us down relative to what robots.txt asks for.
    effective = max(declared, delay if delay is not None else 0.0)
    print(f"robots.txt declares Crawl-Delay: {declared:g}s -> using {effective:g}s")

    paths.RAW.mkdir(parents=True, exist_ok=True)
    last_request = 0.0

    for i, doc in enumerate(docs, 1):
        dest = paths.RAW / f"{doc.doc_id}.pdf"

        if dest.exists() and not force:
            blob = dest.read_bytes()
            doc.sha256 = hashlib.sha256(blob).hexdigest()
            doc.bytes = len(blob)
            doc.fetched_at = doc.fetched_at or datetime.fromtimestamp(
                dest.stat().st_mtime, tz=UTC
            ).isoformat()
            print(f"[{i}/{len(docs)}] cached  {doc.doc_id}")
            continue

        if not robots.allowed(doc.url):
            doc.parse_ok = False
            doc.parse_note = "disallowed by robots.txt"
            print(f"[{i}/{len(docs)}] SKIP    {doc.doc_id} (robots.txt disallows {doc.url})")
            continue

        wait = effective - (time.monotonic() - last_request)
        if wait > 0 and last_request:
            time.sleep(wait)

        try:
            resp = session.get(doc.url, timeout=180)
            last_request = time.monotonic()
            resp.raise_for_status()
            blob = resp.content
        except Exception as exc:  # network/HTTP; recorded, never fatal
            last_request = time.monotonic()
            doc.parse_ok = False
            doc.parse_note = f"fetch failed: {type(exc).__name__}: {exc}"
            print(f"[{i}/{len(docs)}] FAIL    {doc.doc_id}: {exc}")
            continue

        if not blob.startswith(b"%PDF"):
            doc.parse_ok = False
            doc.parse_note = "response was not a PDF"
            print(f"[{i}/{len(docs)}] FAIL    {doc.doc_id}: not a PDF")
            continue

        dest.write_bytes(blob)
        doc.sha256 = hashlib.sha256(blob).hexdigest()
        doc.bytes = len(blob)
        doc.fetched_at = datetime.now(UTC).isoformat()
        print(f"[{i}/{len(docs)}] ok      {doc.doc_id}  {len(blob)/1024:,.0f} KB")

    return docs


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return s


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

# A page yielding fewer characters than this is almost certainly a scan or an
# image-only page. Tracking this matters: bad extraction silently caps retrieval
# quality, and a corpus full of empty text produces an eval that measures nothing.
MIN_CHARS_PER_PAGE = 200


def parse_docs(docs: list[DocRecord], force: bool = False) -> list[DocRecord]:
    """Extract text from each PDF into ``data/processed/<doc_id>.json``.

    Output preserves layout signals the chunker needs:

    * ``page``       - 1-indexed page number, so a citation can point at a page.
    * ``size``       - the dominant font size of the line, used to detect headings.
    * ``bold``       - whether the dominant span is bold, also a heading signal.

    Storing these rather than a flat string is what makes section-aware chunking
    possible later without re-opening the PDFs.
    """

    paths.PROCESSED.mkdir(parents=True, exist_ok=True)

    for i, doc in enumerate(docs, 1):
        src = paths.RAW / f"{doc.doc_id}.pdf"
        dest = paths.PROCESSED / f"{doc.doc_id}.json"

        if not src.exists():
            doc.parse_ok = False
            doc.parse_note = doc.parse_note or "raw pdf missing"
            continue

        if dest.exists() and not force:
            payload = json.loads(dest.read_text(encoding="utf-8"))
        else:
            try:
                payload = _extract(src, doc)
            except Exception as exc:
                doc.parse_ok = False
                doc.parse_note = f"parse failed: {type(exc).__name__}: {exc}"
                print(f"[{i}/{len(docs)}] parse FAIL {doc.doc_id}: {exc}")
                continue
            dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        doc.n_pages = payload["n_pages"]
        doc.n_chars = payload["n_chars"]
        doc.chars_per_page = round(payload["n_chars"] / max(payload["n_pages"], 1), 1)
        doc.parse_ok = doc.chars_per_page >= MIN_CHARS_PER_PAGE
        if not doc.parse_ok:
            doc.parse_note = (
                f"low text yield ({doc.chars_per_page:.0f} chars/page) - "
                "likely a scanned document; excluded from the index"
            )
        print(
            f"[{i}/{len(docs)}] parsed  {doc.doc_id}  "
            f"{payload['n_pages']}p  {doc.chars_per_page:,.0f} chars/page"
            f"{'  <-- LOW YIELD' if not doc.parse_ok else ''}"
        )

    return docs


def _extract(src: Path, doc: DocRecord) -> dict[str, Any]:
    import pymupdf

    lines: list[dict[str, Any]] = []
    n_chars = 0
    with pymupdf.open(src) as pdf:
        n_pages = pdf.page_count
        for pno in range(n_pages):
            page = pdf[pno]
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type") != 0:  # 0 = text, 1 = image
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans)
                    text = _clean(text)
                    if not text:
                        continue
                    # Dominant span = the one contributing the most characters.
                    dom = max(spans, key=lambda s: len(s.get("text", "")), default={})
                    font = (dom.get("font") or "").lower()
                    lines.append(
                        {
                            "page": pno + 1,
                            "t": text,
                            "size": round(float(dom.get("size", 0.0)), 1),
                            "bold": bool("bold" in font or "black" in font),
                        }
                    )
                    n_chars += len(text)

    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "url": doc.url,
        "n_pages": n_pages,
        "n_chars": n_chars,
        "lines": lines,
    }


_WS = re.compile(r"[ \t\xa0]+")


def _clean(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return _WS.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(
    limit: int | None = None,
    delay: float | None = None,
    force_fetch: bool = False,
    force_parse: bool = False,
    skip_fetch: bool = False,
) -> Manifest:
    paths.ensure_dirs()
    session = _session()

    print("reading robots.txt ...")
    robots = Robots(user_agent=USER_AGENT)

    if not robots.allowed(INDEX_URL):
        raise SystemExit(f"robots.txt disallows the index URL {INDEX_URL}; aborting.")

    print("fetching guidance index ...")
    rows = fetch_index(session)
    print(f"index has {len(rows):,} rows")

    docs = select_docs(rows, limit=limit)
    print(f"selected {len(docs)} documents")

    selection = {
        "centers": list(DEFAULT_CENTERS),
        "communication_type": "Guidance Document",
        "final_only": True,
        "topic_pattern": DEFAULT_TOPIC_PATTERN,
        "limit": limit,
        "index_rows_seen": len(rows),
    }

    if not skip_fetch:
        docs = fetch_docs(docs, robots, delay=delay, force=force_fetch, session=session)

    docs = parse_docs(docs, force=force_parse)

    manifest = Manifest(
        created_at=datetime.now(UTC).isoformat(),
        index_url=INDEX_URL,
        selection=selection,
        docs=docs,
    )
    manifest.save()

    ok = sum(1 for d in docs if d.parse_ok)
    low = sum(1 for d in docs if d.parse_ok is False and d.sha256)
    print(
        f"\nmanifest written to {paths.MANIFEST}\n"
        f"  {ok} usable, {low} excluded (low yield / failed), {len(docs)} selected"
    )
    return manifest
