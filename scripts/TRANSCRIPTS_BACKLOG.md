# Earnings-transcript ingestion backlog

Tracks the rolling backfill of earnings-call transcripts via
`scripts/load_transcripts.py` (Alpha Vantage, document-RAG into `filing_chunks`).

## Quota guideline
Alpha Vantage free tier: **25 requests/day, 5 req/min.** Each `(company, quarter)`
is one request. We keep each day's batch to **~20 calls** (margin under 25) and
pace with `--sleep 13`. The loader is **resumable**: it skips `(company, quarter)`
pairs already in the DB *before* making a request, so re-running a list spends
quota only on what's still missing.

## Daily command (latest 2 quarters per company)
```bash
python scripts/load_transcripts.py --tickers <BATCH> --last 2 --sleep 13
```
To check what's already loaded at any time:
```sql
SELECT DISTINCT co.ticker FROM filing_chunks fc
JOIN companies co ON co.id = fc.company_id WHERE fc.form='earnings_call';
```

## Priority list & status

Done (transcripts loaded) — 13 companies, 24 transcripts, 400 chunks as of 2026-10-13:
- AAPL, MSFT, NVDA — 2025 Q1/Q2 + 2026 Q2/Q3
- AMZN, GOOGL, META, TSLA, JPM, JNJ, UNH, XOM — 2026 Q2 (batch 2026-10-13)
- V, WMT — 2026 Q2 + Q3 (batch 2026-10-13; offset fiscal calendars, so Q3 was out)

Pending (carry over — next day, resets the 25/day quota):
- BAC, KO, PEP, HD, PG, DIS, NFLX, MA, CVX, COST  (latest 2 quarters each)

Also re-fetchable later (not yet reported as of 2026-10-13, auto-loads on re-run):
- 2026 Q3 for AMZN, GOOGL, META, TSLA, JPM, JNJ, UNH, XOM — calendar-Q3 calls
  land late Oct–Nov; a later `--last 2` run picks them up (resumable, no waste).

After the pending batch, extend the priority list with the next tranche of the
S&P large caps as needed. Breadth first (latest 2 quarters per company); deepen
history later with `--years/--quarters` once coverage is wide enough.

After the pending batch, extend the priority list with the next tranche of the
S&P large caps as needed. Breadth first (latest 2 quarters per company); deepen
history later with `--years/--quarters` once coverage is wide enough.
