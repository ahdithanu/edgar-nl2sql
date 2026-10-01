"""Load earnings-call transcripts into filing_chunks (document-RAG path).

Sibling to scripts/load_filings.py. Where that loader pulls 10-K Risk Factors +
MD&A from SEC EDGAR, this one pulls earnings-call transcripts from Financial
Modeling Prep (FMP), chunks them, embeds them with Voyage, and stores them in
the SAME filing_chunks table so the existing document-RAG retrieval
(app/filing_retrieval.py) can answer questions like "what did Apple's management
say about margins on the last call?" with no retrieval changes.

WHY reuse filing_chunks instead of a new table: a transcript is just more
narrative text about a company, retrieved the same way as a 10-K section. One
corpus, one HNSW index, one retrieval path. Transcript rows are distinguished by
form='earnings_call'; the quarter is carried in `section` ('earnings_q1'..'q4')
and a synthetic `accession` ('TICKER-YEAR-Qn'), so retrieval can label the
citation "AAPL Earnings Call Q1 FY2024".

PREREQUISITE: an FMP_API_KEY on a plan tier that includes the earnings-call
transcript endpoint (Ultimate+). Without it this script can't fetch anything;
the rest of the system is unaffected (it reads transcripts from the DB).

Idempotent: a transcript's existing chunks (keyed by company+accession) are
deleted before its new ones are written, so re-running refreshes cleanly and a
resumable run skips transcripts already loaded.

CLI:
    python scripts/load_transcripts.py                         # every company, most recent 4 calls each
    python scripts/load_transcripts.py --tickers AAPL,MSFT     # a subset
    python scripts/load_transcripts.py --years 2024,2025 --quarters 1,2,3,4
    python scripts/load_transcripts.py --last 8                # 8 most recent calls per company
    python scripts/load_transcripts.py --dry-run               # fetch + parse, write nothing
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import httpx
import psycopg
import voyageai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402

# FMP endpoints. Base + paths are constants so a plan/version difference is a
# one-line change. Parsing below is deliberately tolerant of shape variants.
FMP_BASE = "https://financialmodelingprep.com"
TRANSCRIPT_PATH = "/api/v3/earning_call_transcript/{symbol}"           # ?year=&quarter=
TRANSCRIPT_DATES_PATH = "/api/v4/earning_call_transcript?symbol={symbol}"  # list of [q, year, date]
MIN_REQUEST_INTERVAL_S = 0.2
MAX_RETRIES = 5

# Transcripts are long (a full call ≈ 8-12k words). Same window as the 10-K
# loader, higher cap: we want the whole call retrievable, still bounded.
CHUNK_CHARS = 3500
CHUNK_OVERLAP = 400
MAX_CHUNKS_PER_CALL = 50

EMBED_MODEL = "voyage-3.5-lite"
EMBED_BATCH = 128

DEFAULT_LAST_N = 4  # most recent calls per company when no explicit years/quarters


# --------------------------------------------------------------------------- #
# Pure helpers (no network / no DB) — unit-tested in tests/test_load_transcripts.py
# --------------------------------------------------------------------------- #

def synthetic_accession(ticker: str, year: int, quarter: int) -> str:
    """Stable synthetic id for a transcript (there is no SEC accession)."""
    return f"{ticker.upper()}-{year}-Q{quarter}"


def section_key(quarter: int) -> str:
    """filing_chunks.section value for a transcript quarter."""
    return f"earnings_q{quarter}"


def normalize_transcript(payload: object) -> dict | None:
    """Normalize an FMP transcript response to {year, quarter, date, content}.

    FMP has returned both a bare object and a single-element list across
    versions, and uses either 'quarter' or 'period' for the quarter. Be
    tolerant: pull the first record and read whichever keys are present.
    Returns None if there's no usable content.
    """
    record = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(record, dict):
        return None
    content = (record.get("content") or "").strip()
    if not content:
        return None

    quarter = record.get("quarter", record.get("period"))
    if isinstance(quarter, str):
        m = re.search(r"\d", quarter)
        quarter = int(m.group()) if m else None
    year = record.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None

    return {
        "year": year,
        "quarter": int(quarter) if quarter is not None else None,
        "date": record.get("date"),
        "content": content,
    }


def normalize_available_dates(payload: object) -> list[tuple[int, int]]:
    """Normalize the 'available dates' response to sorted [(year, quarter), ...].

    FMP returns a list of [quarter, year, date] arrays (or dicts). Returns
    newest-first so callers can take the first N for the most recent calls.
    """
    out: list[tuple[int, int]] = []
    if not isinstance(payload, list):
        return out
    for item in payload:
        try:
            if isinstance(item, dict):
                q = int(item.get("quarter", item.get("period")))
                y = int(item["year"])
            else:  # [quarter, year, date]
                q, y = int(item[0]), int(item[1])
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if 1 <= q <= 4:
            out.append((y, q))
    # newest first
    return sorted(set(out), key=lambda yq: (yq[0], yq[1]), reverse=True)


def chunk_transcript(text: str) -> list[str]:
    """Overlapping character-window chunks, capped per call."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < MAX_CHUNKS_PER_CALL:
        end = start + CHUNK_CHARS
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


# --------------------------------------------------------------------------- #
# FMP client
# --------------------------------------------------------------------------- #

class FMPClient:
    """Rate-limited Financial Modeling Prep client."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise SystemExit(
                "FMP_API_KEY is not set. Earnings transcripts require an FMP plan "
                "tier that includes the earnings-call transcript endpoint (Ultimate+). "
                "Set FMP_API_KEY in your environment and re-run."
            )
        self._key = api_key
        self._client = httpx.Client(timeout=60)
        self._last = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)
        self._last = time.monotonic()

    def _get(self, url: str) -> object:
        sep = "&" if "?" in url else "?"
        full = f"{url}{sep}apikey={self._key}"
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self._client.get(full)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 — retry transient errors
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"GET failed after {MAX_RETRIES} tries: {url}: {last_error}")

    def available(self, symbol: str) -> list[tuple[int, int]]:
        url = FMP_BASE + TRANSCRIPT_DATES_PATH.format(symbol=symbol)
        return normalize_available_dates(self._get(url))

    def transcript(self, symbol: str, year: int, quarter: int) -> dict | None:
        url = FMP_BASE + TRANSCRIPT_PATH.format(symbol=symbol) + f"?year={year}&quarter={quarter}"
        return normalize_transcript(self._get(url))


# --------------------------------------------------------------------------- #
# DB helpers + orchestration
# --------------------------------------------------------------------------- #

def embed_documents(texts: list[str], api_key: str) -> list[list[float]]:
    client = voyageai.Client(api_key=api_key)
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        out.extend(client.embed(batch, model=EMBED_MODEL, input_type="document").embeddings)
    return out


def db_companies(conn: psycopg.Connection, tickers: list[str] | None, limit: int | None):
    sql = "SELECT id, ticker FROM companies"
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


def existing_accessions(conn: psycopg.Connection) -> set[str]:
    """Already-loaded transcript ids, so a re-run skips them (resumable)."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT accession FROM filing_chunks WHERE form = 'earnings_call'")
        return {r[0] if not isinstance(r, dict) else r["accession"] for r in cur.fetchall()}


def _targets(
    client: FMPClient, ticker: str, years: list[int] | None,
    quarters: list[int] | None, last_n: int,
) -> list[tuple[int, int]]:
    """Resolve which (year, quarter) calls to load for one company."""
    if years and quarters:
        return [(y, q) for y in years for q in quarters]
    return client.available(ticker)[:last_n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Load earnings-call transcripts into filing_chunks.")
    parser.add_argument("--tickers", help="Comma-separated subset (default: all companies in the DB)")
    parser.add_argument("--limit", type=int, help="Only the first N companies")
    parser.add_argument("--years", help="Comma-separated years (with --quarters)")
    parser.add_argument("--quarters", help="Comma-separated quarters 1-4 (with --years)")
    parser.add_argument("--last", type=int, default=DEFAULT_LAST_N,
                        help=f"Most recent N calls per company when years/quarters omitted (default {DEFAULT_LAST_N})")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + parse, write nothing")
    parser.add_argument("--force", action="store_true", help="Reload transcripts already present")
    args = parser.parse_args()

    settings = get_settings()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    years = [int(y) for y in args.years.split(",")] if args.years else None
    quarters = [int(q) for q in args.quarters.split(",")] if args.quarters else None

    client = FMPClient(settings.fmp_api_key)

    with psycopg.connect(settings.database_url) as conn:
        companies = db_companies(conn, tickers, args.limit)
        loaded = existing_accessions(conn)

    print(f"Loading earnings transcripts for {len(companies)} companies"
          f"{' [dry run]' if args.dry_run else ''}")

    calls = skipped = failed = 0
    for company_id, ticker in companies:
        try:
            targets = _targets(client, ticker, years, quarters, args.last)
        except Exception as exc:  # noqa: BLE001 — one company must not sink the run
            print(f"  {ticker}: date lookup FAILED ({exc})", file=sys.stderr)
            failed += 1
            continue

        for year, quarter in targets:
            acc = synthetic_accession(ticker, year, quarter)
            if acc in loaded and not args.force:
                skipped += 1
                continue
            try:
                record = client.transcript(ticker, year, quarter)
            except Exception as exc:  # noqa: BLE001
                print(f"  {acc}: FETCH FAILED ({exc})", file=sys.stderr)
                failed += 1
                continue
            if not record or not record["content"]:
                continue

            chunks = chunk_transcript(record["content"])
            print(f"  {acc}: {len(chunks)} chunks", flush=True)
            if args.dry_run or not chunks:
                continue

            embeddings = embed_documents(chunks, settings.voyage_api_key)
            section = section_key(quarter)
            try:
                with psycopg.connect(settings.database_url) as wconn, wconn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM filing_chunks WHERE company_id = %s AND accession = %s",
                        (company_id, acc),
                    )
                    for idx, (content, emb) in enumerate(zip(chunks, embeddings)):
                        vec = "[" + ",".join(str(v) for v in emb) + "]"
                        cur.execute(
                            "INSERT INTO filing_chunks (company_id, accession, form, fiscal_year, "
                            "section, chunk_index, content, embedding) "
                            "VALUES (%s,%s,'earnings_call',%s,%s,%s,%s,%s::vector)",
                            (company_id, acc, year, section, idx, content, vec),
                        )
                    wconn.commit()
                calls += 1
            except Exception as exc:  # noqa: BLE001 — a write failure must not sink the run
                print(f"  {acc}: WRITE FAILED ({exc})", file=sys.stderr)
                failed += 1

    print(f"\nLoaded {calls} transcripts, skipped {skipped} already-present, failed {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
