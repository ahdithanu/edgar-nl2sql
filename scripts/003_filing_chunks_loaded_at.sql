-- Add a load timestamp to filing_chunks so "what was ingested recently" is a
-- plain SQL question (the weekly transcript refresh and backfills otherwise
-- leave no queryable trace of WHEN a chunk arrived).
--
-- NOT NULL DEFAULT now(): new rows (10-K sections and earnings-call transcripts
-- alike) get their load time automatically on INSERT — no loader change needed.
-- The loaders delete-then-insert per document, so a re-load refreshes loaded_at
-- to the latest load, which is the intended meaning.
--
-- One-time backfill caveat: existing rows are stamped with the migration time,
-- not their true original load time. That is only approximate for the older
-- 10-K chunks; it is accurate for everything loaded from here on.
--
-- Idempotent. Apply as a privileged role (Supabase SQL Editor / postgres).

ALTER TABLE filing_chunks
    ADD COLUMN IF NOT EXISTS loaded_at TIMESTAMPTZ NOT NULL DEFAULT now();

-- Index for recency queries (e.g. "loaded in the last 7 days", newest first).
CREATE INDEX IF NOT EXISTS idx_filing_chunks_loaded_at
    ON filing_chunks (loaded_at DESC);

-- ============================================================================
-- Example: transcripts newly loaded in the last 7 days, by transcript.
--   SELECT co.ticker, fc.accession, min(fc.loaded_at) AS loaded_at, count(*) AS chunks
--   FROM filing_chunks fc JOIN companies co ON co.id = fc.company_id
--   WHERE fc.form = 'earnings_call' AND fc.loaded_at >= now() - interval '7 days'
--   GROUP BY co.ticker, fc.accession
--   ORDER BY loaded_at DESC;
-- ============================================================================
