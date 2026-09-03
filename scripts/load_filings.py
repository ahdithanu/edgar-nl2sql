"""Load 10-K narrative text (Risk Factors + MD&A) into filing_chunks.

This is the document-RAG half of the system, parallel to the structured
financial_metrics loader. For each company already in the database it fetches
the latest 10-K from SEC EDGAR, extracts Item 1A (Risk Factors) and Item 7
(MD&A), splits each into overlapping chunks, embeds them with Voyage, and
upserts them into filing_chunks. Qualitative questions ("what risks did Apple
cite?", "how did management explain the margin change?") retrieve from here.

WHY only two sections, capped: a full 10-K is 100-300 pages; Risk Factors and
MD&A carry nearly all the answerable qualitative content, and capping chunks
per section keeps the embedding volume bounded across the whole universe while
front-loading the most important discussion.

Idempotent: a company's existing chunks are deleted before its new ones are
written, so re-running refreshes cleanly.

CLI:
    python scripts/load_filings.py                       # every company in the DB
    python scripts/load_filings.py --tickers AAPL,JPM    # a subset
    python scripts/load_filings.py --limit 20            # first N companies
    python scripts/load_filings.py --dry-run             # fetch + parse, no writes
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import psycopg
import voyageai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402

SEC_USER_AGENT = "edgar-nl2sql portfolio project ahdi@uaconsulting.co"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"
MIN_REQUEST_INTERVAL_S = 0.15
MAX_RETRIES = 5

# Chunking. ~3200 chars ≈ 800 tokens; overlap keeps a sentence that straddles a
# boundary retrievable. The per-section cap bounds total embedding volume across
# ~478 companies (worst case ≈ 478 * 2 * cap chunks) and front-loads the most
# important discussion, since Risk Factors and MD&A lead with their summaries.
CHUNK_CHARS = 3200
CHUNK_OVERLAP = 400
MAX_CHUNKS_PER_SECTION = 18

EMBED_MODEL = "voyage-3.5-lite"
EMBED_BATCH = 128

SECTIONS = {
    # section key -> (start pattern, end-boundary patterns)
    "risk_factors": (
        r"item\s*1a\.?\s*risk\s*factors",
        [r"item\s*1b\b", r"item\s*2\b", r"item\s*3\b"],
    ),
    "mda": (
        r"item\s*7\.?\s*management.s\s*discussion",
        [r"item\s*7a\b", r"item\s*8\b"],
    ),
}


@dataclass
class Filing:
    accession: str
    primary_doc: str
    fiscal_year: int | None


class EdgarClient:
    """Rate-limited SEC client honoring the fair-access rules (UA + <=10 rps)."""

    def __init__(self) -> None:
        self._client = httpx.Client(headers={"User-Agent": SEC_USER_AGENT}, timeout=90)
        self._last = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)
        self._last = time.monotonic()

    def _get(self, url: str) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self._client.get(url)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001 — retry transient errors
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"GET failed after {MAX_RETRIES} tries: {url}: {last_error}")

    def get_json(self, url: str) -> dict:
        return self._get(url).json()

    def get_text(self, url: str) -> str:
        return self._get(url).text


def latest_10k(client: EdgarClient, cik10: str) -> Filing | None:
    """The most recent 10-K for a CIK, or None if the company has never filed one."""
    data = client.get_json(SUBMISSIONS_URL.format(cik=cik10))
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            report_date = recent.get("reportDate", [None] * len(recent["form"]))[i]
            fy = int(report_date[:4]) if report_date else None
            return Filing(recent["accessionNumber"][i], recent["primaryDocument"][i], fy)
    return None


def html_to_text(html: str) -> str:
    """Strip a filing's HTML to readable text, keeping paragraph breaks."""
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"&#160;|&nbsp;", " ", html)
    html = re.sub(r"&#8217;|&#8216;|&#x2019;|&#x2018;", "'", html)
    html = re.sub(r"&#8220;|&#8221;|&#x201c;|&#x201d;", '"', html)
    html = re.sub(r"&#8212;|&#8211;", "-", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n\s*\n+", "\n", html)
    return html


def extract_section(text: str, start_pat: str, end_pats: list[str], min_len: int = 2500) -> str | None:
    """Return the body of a 10-K item.

    A 10-K mentions "Item 1A" and "Item 7" several times (table of contents,
    cross-references, then the real section). We take every start match and pick
    the one whose span to the next real end boundary is LONGEST: TOC/reference
    occurrences are short, the actual section body is large. This is what makes
    extraction robust across filers that format their headings differently.
    """
    starts = [m.start() for m in re.finditer(start_pat, text, re.I)]
    best: str | None = None
    for s in starts:
        tail = text[s:]
        ends = sorted(
            e + 80
            for p in end_pats
            for e in [m.start() for m in re.finditer(p, tail[80:], re.I)]
        )
        candidate_ends = [e for e in ends if e >= min_len]
        end = candidate_ends[0] if candidate_ends else min(len(tail), 80000)
        span = tail[:end].strip()
        if best is None or len(span) > len(best):
            best = span
    return best if best and len(best) >= min_len else None


def chunk_text(text: str) -> list[str]:
    """Overlapping character-window chunks, capped per section."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < MAX_CHUNKS_PER_SECTION:
        end = start + CHUNK_CHARS
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def embed_documents(texts: list[str], api_key: str) -> list[list[float]]:
    client = voyageai.Client(api_key=api_key)
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        out.extend(client.embed(batch, model=EMBED_MODEL, input_type="document").embeddings)
    return out


def db_companies(conn: psycopg.Connection, tickers: list[str] | None, limit: int | None):
    """(company_id, ticker, cik) for the companies to load, from the DB."""
    sql = "SELECT id, ticker, cik FROM companies"
    params: list = []
    if tickers:
        sql += " WHERE ticker = ANY(%s)"
        params.append(tickers)
    sql += " ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def process_company(client: EdgarClient, cik10: str) -> dict[str, list[str]]:
    """Fetch the latest 10-K and return {section: [chunks]} (may be partial)."""
    filing = latest_10k(client, cik10)
    if filing is None:
        return {}
    cik_int = str(int(cik10))
    url = ARCHIVE_URL.format(
        cik_int=cik_int, acc=filing.accession.replace("-", ""), doc=filing.primary_doc
    )
    text = html_to_text(client.get_text(url))
    result: dict[str, list[str]] = {"_filing": filing}  # type: ignore[dict-item]
    for section, (start_pat, end_pats) in SECTIONS.items():
        body = extract_section(text, start_pat, end_pats)
        if body:
            result[section] = chunk_text(body)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Load 10-K narrative text into filing_chunks.")
    parser.add_argument("--tickers", help="Comma-separated subset (default: all companies in the DB)")
    parser.add_argument("--limit", type=int, help="Only the first N companies")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + parse, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="Reload even companies that already have chunks (default: skip them)")
    args = parser.parse_args()

    settings = get_settings()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    client = EdgarClient()

    # Resolve the company list and the already-loaded set with a short-lived
    # connection, then CLOSE it. WHY: the loop below spends most of its time on
    # slow SEC fetches and embedding calls; holding one DB connection open
    # across all of that let Supabase's pooler drop it mid-run (the first
    # attempt died at ~115/478 with an SSL SYSCALL error). We instead open a
    # fresh short connection only for each company's write.
    with psycopg.connect(settings.database_url) as conn:
        companies = db_companies(conn, tickers, args.limit)
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT company_id FROM filing_chunks")
            loaded = {r[0] if not isinstance(r, dict) else r["company_id"] for r in cur.fetchall()}

    if not args.force and not tickers:
        # Resumable: skip companies already loaded, so a re-run finishes the rest.
        companies = [c for c in companies if c[0] not in loaded]

    print(f"Loading 10-K narrative for {len(companies)} companies "
          f"({len(loaded)} already loaded){' [dry run]' if args.dry_run else ''}")

    both = one = none = failed = 0
    for company_id, ticker, cik in companies:
        try:
            sections = process_company(client, cik)
        except Exception as exc:  # noqa: BLE001 — one bad filing must not sink the run
            print(f"  {ticker}: FAILED ({exc})", file=sys.stderr)
            failed += 1
            continue

        filing: Filing | None = sections.pop("_filing", None)  # type: ignore[assignment]
        chunk_rows = [
            (section, idx, content)
            for section, chunks in sections.items()
            for idx, content in enumerate(chunks)
        ]
        got = set(sections)
        both += got == set(SECTIONS)
        one += len(got) == 1
        none += len(got) == 0
        print(f"  {ticker}: {', '.join(f'{s}={len(sections[s])}' for s in sections) or 'no sections'}"
              f" ({len(chunk_rows)} chunks)", flush=True)

        if args.dry_run or not chunk_rows or filing is None:
            continue

        embeddings = embed_documents([c for _, _, c in chunk_rows], settings.voyage_api_key)
        try:
            # Fresh connection per company: robust to the pooler dropping an idle
            # connection during the slow fetch/embed above.
            with psycopg.connect(settings.database_url) as wconn, wconn.cursor() as cur:
                cur.execute("DELETE FROM filing_chunks WHERE company_id = %s", (company_id,))
                for (section, idx, content), emb in zip(chunk_rows, embeddings):
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    cur.execute(
                        "INSERT INTO filing_chunks (company_id, accession, form, fiscal_year, "
                        "section, chunk_index, content, embedding) "
                        "VALUES (%s,%s,'10-K',%s,%s,%s,%s,%s::vector)",
                        (company_id, filing.accession, filing.fiscal_year, section, idx, content, vec),
                    )
                wconn.commit()
        except Exception as exc:  # noqa: BLE001 — a write failure must not sink the run
            print(f"  {ticker}: WRITE FAILED ({exc})", file=sys.stderr)
            failed += 1

    total = len(companies)
    print(f"\nProcessed {total}: both sections {both}, one {one}, none {none}, failed {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
