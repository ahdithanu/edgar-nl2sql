#!/usr/bin/env python3
"""Load SEC EDGAR XBRL company facts into Postgres for the NL->SQL RAG system.

Why this script exists
----------------------
The RAG pipeline answers financial questions by generating SQL against a small,
well-understood star schema (companies / filings / financial_metrics). This
script is the single source of that data: it pulls raw XBRL "company facts"
from SEC EDGAR for 25 large-cap companies, normalizes them into five canonical
metrics, and upserts them idempotently so the loader can be re-run at any time
without duplicating rows.

The tricky part (worth reading closely)
---------------------------------------
EDGAR company-facts JSON tags every fact with the *filing's* fiscal year and
period (`fy`/`fp`) — including prior-year comparative figures restated in that
filing. A 10-K for FY2023 therefore contains revenue rows for FY2023, FY2022
and FY2021 that are ALL labeled fy=2023/fp=FY. Naively keying on (fy, fp)
would pick an arbitrary comparative. We disambiguate with two filters:

  1. Duration windows: annual flow facts must span ~a year (330-380 days) and
     come from a 10-K; quarterly flow facts must span ~a quarter (70-100 days)
     and come from a 10-Q. This drops year-to-date figures (e.g. the 9-month
     cumulative number in a Q3 10-Q).
  2. Latest `end` wins per (fy, fp) group: among facts sharing a fiscal label,
     the one with the most recent period-end date is the current period; the
     earlier ones are the comparatives.

EDGAR also never reports a standalone Q4 for flow metrics (Q4 lives implicitly
inside the 10-K's full-year figure), so we derive Q4 = FY - (Q1+Q2+Q3). That
subtraction is exact for dollar flows and only approximate for EPS (share
counts differ across quarters) — the context corpus documents that caveat so
generated SQL and answers can acknowledge it.

Usage:
    python scripts/load_edgar.py [--tickers AAPL,MSFT] [--dry-run]

SEC fair-access rules honored here: a descriptive User-Agent on every request
and <=10 requests/second (we sleep 0.15s between calls, and back off on
429/5xx responses).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Scripts are run as `python scripts/load_edgar.py` from the repo root, which
# puts scripts/ (not the root) on sys.path. Bootstrap the root so `app.*`
# imports resolve without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402

import psycopg  # noqa: E402

# --------------------------------------------------------------------------
# Constants (from CONTRACTS.md — do not tweak casually; eval depends on them)
# --------------------------------------------------------------------------

DEFAULT_TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "BAC", "GS", "V", "MA",
    "WMT", "COST", "HD", "KO", "PEP", "MCD",
    "XOM", "CVX", "JNJ", "PFE", "UNH", "DIS", "NFLX",
]

# SEC requires a User-Agent identifying the requester on EVERY request.
SEC_USER_AGENT = "edgar-nl2sql portfolio project ahdi@uaconsulting.co"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# <=10 req/s per SEC fair-access policy; 0.15s spacing keeps us safely under.
MIN_REQUEST_INTERVAL_S = 0.15
MAX_RETRIES = 5

FISCAL_YEAR_MIN = 2020
FISCAL_YEAR_MAX = 2024

# us-gaap tag fallbacks: companies report under different tags depending on
# era and industry — and can switch tags mid-history. Fallback is applied per
# (fiscal_year, fiscal_period): the first tag with a usable fact for a period
# wins, and later tags fill in periods the earlier tags don't cover.
METRIC_TAGS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}
METRICS: list[str] = list(METRIC_TAGS)

# Flow metrics accumulate over a period (have start+end); instants are
# balance-sheet snapshots (end only). This distinction drives period selection
# and the Q4 derivation below.
FLOW_METRICS = {"revenue", "net_income", "eps_diluted"}
INSTANT_METRICS = {"total_assets", "total_liabilities"}

# (EDGAR units key, unit string we store)
METRIC_UNIT: dict[str, tuple[str, str]] = {
    "revenue": ("USD", "USD"),
    "net_income": ("USD", "USD"),
    "total_assets": ("USD", "USD"),
    "total_liabilities": ("USD", "USD"),
    "eps_diluted": ("USD/shares", "USD/share"),
}

ANNUAL_DURATION_DAYS = (330, 380)     # a fiscal year, with slack for 52/53-week calendars
QUARTERLY_DURATION_DAYS = (70, 100)   # a fiscal quarter, with slack


# --------------------------------------------------------------------------
# EDGAR HTTP client: rate limiting + backoff
# --------------------------------------------------------------------------

class EdgarClient:
    """Thin httpx wrapper enforcing SEC fair-access rules.

    Why a class: the rate limiter needs state (timestamp of the last request)
    shared across every call, and we want one connection pool for all ~51
    requests (1 ticker map + 2 per company).
    """

    def __init__(self) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": SEC_USER_AGENT},  # on EVERY request
            timeout=30.0,
            follow_redirects=True,
        )
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        """Sleep just enough to stay under 10 req/s."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)
        self._last_request_at = time.monotonic()

    def get_json(self, url: str) -> dict:
        """GET a JSON document, retrying 429/5xx with exponential backoff.

        4xx errors other than 429 are permanent (bad CIK, moved endpoint) and
        raise immediately — retrying would just burn our request budget.
        """
        last_error = ""
        for attempt in range(MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self._client.get(url)
            except httpx.HTTPError as exc:
                # Transient network failure: treat like a retryable status.
                last_error = f"network error: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                else:
                    raise RuntimeError(f"EDGAR returned HTTP {resp.status_code} for {url}")
            if attempt < MAX_RETRIES:
                backoff = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"  retrying {url} in {backoff}s ({last_error})", file=sys.stderr)
                time.sleep(backoff)
        raise RuntimeError(f"EDGAR request failed after {MAX_RETRIES + 1} tries: {url} ({last_error})")

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------
# Data shapes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    """One normalized XBRL fact after basic field validation."""

    fiscal_year: int
    fiscal_period: str           # 'FY','Q1','Q2','Q3' (Q4 only via derivation)
    value: float
    start: dt.date | None        # None for instant (balance-sheet) facts
    end: dt.date
    accession: str
    form: str                    # normalized: '10-K' or '10-Q' (amendments folded in)
    filed: dt.date


@dataclass
class MetricRow:
    """One row destined for financial_metrics."""

    metric: str
    fiscal_year: int
    fiscal_period: str
    value: float
    unit: str
    start: dt.date | None
    end: dt.date
    accession: str               # which filing this row attaches to


@dataclass
class FilingInfo:
    form: str
    filed: dt.date
    fiscal_year: int
    fiscal_period: str


@dataclass
class CompanyLoad:
    """Everything we intend to upsert for one company."""

    ticker: str
    cik: str                     # zero-padded 10 digits
    name: str
    sic_description: str | None
    filings: dict[str, FilingInfo] = field(default_factory=dict)   # accession -> info
    rows: list[MetricRow] = field(default_factory=list)


# --------------------------------------------------------------------------
# EDGAR parsing
# --------------------------------------------------------------------------

def _parse_date(raw: str | None) -> dt.date | None:
    return dt.date.fromisoformat(raw) if raw else None


def _normalize_form(form: str) -> str | None:
    """Fold amendments ('10-K/A') into their base form; reject everything else.

    The schema documents filings.form as '10-K' | '10-Q', and amendments carry
    the corrected numbers we actually want, so we keep them under the base form.
    """
    base = form.split("/")[0].strip()
    return base if base in ("10-K", "10-Q") else None


def extract_facts(company_facts: dict, metric: str, tag: str) -> list[Fact]:
    """Pull raw facts for one metric from ONE us-gaap tag.

    Tag fallback happens per period in select_metric_periods, not here: a
    company can switch tags mid-history (e.g. NVDA reported revenue under
    RevenueFromContractWithCustomerExcludingAssessedTax through FY2022, then
    under plain Revenues from FY2023), so "first tag with any facts wins for
    the whole company" would silently drop the years living under later tags.
    """
    gaap = company_facts.get("facts", {}).get("us-gaap", {})
    units_key, _ = METRIC_UNIT[metric]

    entries = gaap.get(tag, {}).get("units", {}).get(units_key, [])
    facts: list[Fact] = []
    for e in entries:
        form = _normalize_form(e.get("form") or "")
        fy = e.get("fy")
        fp = e.get("fp")
        end = _parse_date(e.get("end"))
        filed = _parse_date(e.get("filed"))
        val = e.get("val")
        if form is None or fy is None or fp is None or end is None or filed is None or val is None:
            continue
        facts.append(
            Fact(
                fiscal_year=int(fy),
                fiscal_period=str(fp),
                value=float(val),
                start=_parse_date(e.get("start")),
                end=end,
                accession=str(e["accn"]),
                form=form,
                filed=filed,
            )
        )
    return facts


def select_periods(facts: list[Fact], metric: str) -> dict[tuple[int, str], Fact]:
    """Reduce raw facts to exactly one fact per (fiscal_year, fiscal_period).

    This is the comparative-period dedupe described in the module docstring:
    filter to plausibly-current facts (right form + right duration window for
    flows), then keep the fact with the LATEST end date in each (fy, fp)
    group — comparatives restated in later filings always have earlier ends.
    """
    selected: dict[tuple[int, str], Fact] = {}
    is_flow = metric in FLOW_METRICS

    for f in facts:
        if not (FISCAL_YEAR_MIN <= f.fiscal_year <= FISCAL_YEAR_MAX):
            continue

        if is_flow:
            # Flows need a duration; facts missing `start` are unusable.
            if f.start is None:
                continue
            duration = (f.end - f.start).days
            if f.fiscal_period == "FY" and f.form == "10-K":
                lo, hi = ANNUAL_DURATION_DAYS
            elif f.fiscal_period in ("Q1", "Q2", "Q3") and f.form == "10-Q":
                lo, hi = QUARTERLY_DURATION_DAYS
            else:
                continue  # Q4 flows never come from EDGAR directly; derived below
            if not (lo <= duration <= hi):
                continue  # e.g. a 9-month YTD figure inside a Q3 10-Q
        else:
            # Instants: no duration to check; FY snapshot comes from the 10-K,
            # Q1-Q3 snapshots from 10-Qs. (There is no separate Q4 row: the FY
            # row IS the fiscal-year-end balance.)
            if f.fiscal_period == "FY" and f.form == "10-K":
                pass
            elif f.fiscal_period in ("Q1", "Q2", "Q3") and f.form == "10-Q":
                pass
            else:
                continue

        key = (f.fiscal_year, f.fiscal_period)
        incumbent = selected.get(key)
        # Latest end wins: the current period beats comparatives. On a tie
        # (same period end) the latest FILED wins: an amendment (10-K/A,
        # 10-Q/A) covers the same period as the original filing but carries
        # the corrected, restated numbers — exactly the values we want.
        if incumbent is None or (f.end, f.filed) > (incumbent.end, incumbent.filed):
            selected[key] = f

    return selected


def select_metric_periods(company_facts: dict, metric: str) -> dict[tuple[int, str], Fact]:
    """One fact per (fiscal_year, fiscal_period), applying tag fallback PER PERIOD.

    Each tag's facts go through select_periods independently, then merge with
    earlier tags taking precedence per period. This keeps "first tag wins"
    semantics where tags overlap, while still filling in periods a company
    only ever reported under a later fallback tag (see extract_facts).
    """
    selected: dict[tuple[int, str], Fact] = {}
    for tag in METRIC_TAGS[metric]:
        for key, fact in select_periods(extract_facts(company_facts, metric, tag), metric).items():
            selected.setdefault(key, fact)  # earlier (more specific) tag wins per period
    return selected


def derive_q4(selected: dict[tuple[int, str], Fact], metric: str, ticker: str) -> list[MetricRow]:
    """Derive Q4 flow rows as FY - (Q1+Q2+Q3).

    EDGAR has no standalone Q4 flow facts (companies file a 10-K, not a fourth
    10-Q), so without this step every "compare quarters" question would
    silently miss a quarter. The derived row is attached to the 10-K filing.
    NOTE: for eps_diluted the subtraction is approximate (quarterly EPS uses
    quarterly share counts) — documented in the RAG glossary so answers can
    carry the caveat.
    """
    if metric not in FLOW_METRICS:
        return []
    _, unit = METRIC_UNIT[metric]
    rows: list[MetricRow] = []
    fys = sorted({fy for (fy, fp) in selected if fp == "FY"})
    for fy in fys:
        fy_fact = selected[(fy, "FY")]
        quarters = [selected.get((fy, q)) for q in ("Q1", "Q2", "Q3")]
        if any(q is None for q in quarters):
            missing = [q for q, f in zip(("Q1", "Q2", "Q3"), quarters) if f is None]
            print(
                f"  WARNING: {ticker} {metric} FY{fy}: cannot derive Q4 "
                f"(missing {','.join(missing)})",
                file=sys.stderr,
            )
            continue
        q1, q2, q3 = quarters  # type: ignore[misc]  # narrowed by the check above
        rows.append(
            MetricRow(
                metric=metric,
                fiscal_year=fy,
                fiscal_period="Q4",
                value=fy_fact.value - (q1.value + q2.value + q3.value),
                unit=unit,
                # Q4 spans from just after Q3's end to fiscal year end.
                start=q3.end + dt.timedelta(days=1),
                end=fy_fact.end,
                accession=fy_fact.accession,  # attach to the 10-K per contract
            )
        )
    return rows


# --------------------------------------------------------------------------
# Per-company orchestration
# --------------------------------------------------------------------------

def fetch_cik_map(client: EdgarClient) -> dict[str, tuple[str, str]]:
    """ticker -> (zero-padded 10-digit CIK, company title)."""
    raw = client.get_json(TICKER_MAP_URL)
    # File shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    return {
        entry["ticker"].upper(): (f"{int(entry['cik_str']):010d}", entry["title"])
        for entry in raw.values()
    }


def process_company(client: EdgarClient, ticker: str, cik: str, name: str) -> CompanyLoad:
    """Fetch + normalize everything for one company (no DB access here).

    Keeping fetch/transform separate from DB writes makes --dry-run trivial
    and keeps each phase independently testable.
    """
    sic = client.get_json(SUBMISSIONS_URL.format(cik=cik)).get("sicDescription") or None
    facts_json = client.get_json(COMPANY_FACTS_URL.format(cik=cik))

    load = CompanyLoad(ticker=ticker, cik=cik, name=name, sic_description=sic)

    for metric in METRICS:
        selected = select_metric_periods(facts_json, metric)
        if not selected:
            print(f"  WARNING: {ticker}: no usable facts for {metric}", file=sys.stderr)
            continue
        _, unit = METRIC_UNIT[metric]

        for (fy, fp), fact in sorted(selected.items()):
            load.rows.append(
                MetricRow(
                    metric=metric,
                    fiscal_year=fy,
                    fiscal_period=fp,
                    value=fact.value,
                    unit=unit,
                    start=fact.start if metric in FLOW_METRICS else None,
                    end=fact.end,
                    accession=fact.accession,
                )
            )
            # Record the filing each kept fact came from (first sighting wins;
            # every fact from one accession shares the same form/filed/fy/fp).
            load.filings.setdefault(
                fact.accession,
                FilingInfo(form=fact.form, filed=fact.filed, fiscal_year=fy, fiscal_period=fp),
            )

        load.rows.extend(derive_q4(selected, metric, ticker))

    return load


# --------------------------------------------------------------------------
# Database writes (idempotent upserts)
# --------------------------------------------------------------------------
# We open our own psycopg connection (from settings.database_url) instead of
# borrowing app.db.get_pool(): the pool's row factory is an app-internal
# detail, and a loader script wants plain tuples + explicit transactions.

def write_company(conn: psycopg.Connection, load: CompanyLoad) -> None:
    """Upsert one company's data inside a single transaction.

    ON CONFLICT everywhere means re-running the loader (or resuming after a
    crash) converges to the same state instead of duplicating rows.
    """
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO companies (cik, ticker, name, sic_description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (cik) DO UPDATE
                SET ticker = EXCLUDED.ticker,
                    name = EXCLUDED.name,
                    sic_description = EXCLUDED.sic_description
            RETURNING id
            """,
            (load.cik, load.ticker, load.name, load.sic_description),
        )
        row = cur.fetchone()
        assert row is not None  # RETURNING always yields a row on INSERT/UPDATE
        company_id: int = row[0]

        filing_ids: dict[str, int] = {}
        for accession, info in sorted(load.filings.items()):
            cur.execute(
                """
                INSERT INTO filings
                    (company_id, accession_number, form, filed_date, fiscal_year, fiscal_period)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO UPDATE
                    SET filed_date = EXCLUDED.filed_date
                RETURNING id
                """,
                (company_id, accession, info.form, info.filed, info.fiscal_year, info.fiscal_period),
            )
            row = cur.fetchone()
            assert row is not None
            filing_ids[accession] = row[0]

        for m in load.rows:
            cur.execute(
                """
                INSERT INTO financial_metrics
                    (company_id, filing_id, metric, fiscal_year, fiscal_period,
                     value, unit, start_date, end_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (company_id, metric, fiscal_year, fiscal_period) DO UPDATE
                    SET value = EXCLUDED.value,
                        unit = EXCLUDED.unit,
                        filing_id = EXCLUDED.filing_id,
                        start_date = EXCLUDED.start_date,
                        end_date = EXCLUDED.end_date
                """,
                (
                    company_id,
                    filing_ids.get(m.accession),
                    m.metric,
                    m.fiscal_year,
                    m.fiscal_period,
                    m.value,
                    m.unit,
                    m.start,
                    m.end,
                ),
            )


# --------------------------------------------------------------------------
# Summary + CLI
# --------------------------------------------------------------------------

def print_summary(loads: list[CompanyLoad], failed: list[str], dry_run: bool) -> None:
    """Print a company x metric row-count matrix — the loader's receipt."""
    header = ["ticker"] + METRICS + ["total"]
    widths = [max(8, len(h) + 2) for h in header]

    def fmt(cells: list[str]) -> str:
        return "".join(c.ljust(w) for c, w in zip(cells, widths))

    title = "LOAD SUMMARY (dry run — nothing written)" if dry_run else "LOAD SUMMARY"
    print(f"\n{title}")
    print(fmt(header))
    print("-" * sum(widths))

    grand_total = 0
    for load in loads:
        counts = {m: 0 for m in METRICS}
        for r in load.rows:
            counts[r.metric] += 1
        total = sum(counts.values())
        grand_total += total
        print(fmt([load.ticker] + [str(counts[m]) for m in METRICS] + [str(total)]))

    print("-" * sum(widths))
    print(f"{len(loads)} companies, {grand_total} metric rows")
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load SEC EDGAR facts into Postgres.")
    parser.add_argument(
        "--tickers",
        help="Comma-separated subset to load (default: all 25 contract tickers)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and process, print the summary, but write nothing to the DB",
    )
    args = parser.parse_args()

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else DEFAULT_TICKERS
    )

    client = EdgarClient()
    loads: list[CompanyLoad] = []
    failed: list[str] = []

    try:
        print(f"Fetching CIK map for {len(tickers)} tickers...")
        cik_map = fetch_cik_map(client)

        for ticker in tickers:
            if ticker not in cik_map:
                print(f"WARNING: {ticker} not found in SEC ticker map — skipped", file=sys.stderr)
                failed.append(ticker)
                continue
            cik, name = cik_map[ticker]
            print(f"Loading {ticker} (CIK {cik})...")
            try:
                loads.append(process_company(client, ticker, cik, name))
            except Exception as exc:  # keep going: one bad company shouldn't sink the batch
                print(f"ERROR: {ticker} failed: {exc}", file=sys.stderr)
                failed.append(ticker)
    finally:
        client.close()

    if not args.dry_run and loads:
        settings = get_settings()
        with psycopg.connect(settings.database_url) as conn:
            for load in loads:
                write_company(conn, load)  # one transaction per company

    print_summary(loads, failed, args.dry_run)

    # Nonzero exit on hard failure so CI / operators notice.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
