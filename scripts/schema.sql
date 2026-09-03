-- edgar-nl2sql database schema (idempotent — safe to re-run).
--
-- One Postgres database holds BOTH the relational financial data (what generated SQL
-- queries against) AND the RAG corpus embeddings (what retrieval searches). Keeping them
-- together means one connection pool, one backup story, and pgvector similarity search
-- with plain SQL — no separate vector store to operate.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS companies (
    id               SERIAL PRIMARY KEY,
    cik              TEXT NOT NULL UNIQUE,      -- zero-padded 10-digit SEC CIK
    ticker           TEXT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    sic_description  TEXT
);

CREATE TABLE IF NOT EXISTS filings (
    id                SERIAL PRIMARY KEY,
    company_id        INT NOT NULL REFERENCES companies(id),
    accession_number  TEXT NOT NULL UNIQUE,
    form              TEXT NOT NULL,            -- '10-K' | '10-Q'
    filed_date        DATE NOT NULL,
    fiscal_year       INT NOT NULL,
    fiscal_period     TEXT NOT NULL             -- 'FY','Q1','Q2','Q3','Q4'
);

-- Long/narrow metric layout (one row per company+metric+period) rather than wide columns:
-- generated SQL stays simple (filter on `metric`), and adding a metric is a data change,
-- not a schema migration.
CREATE TABLE IF NOT EXISTS financial_metrics (
    id             SERIAL PRIMARY KEY,
    company_id     INT NOT NULL REFERENCES companies(id),
    filing_id      INT REFERENCES filings(id),
    metric         TEXT NOT NULL,               -- 'revenue','net_income','total_assets','total_liabilities','eps_diluted'
    fiscal_year    INT NOT NULL,
    fiscal_period  TEXT NOT NULL,               -- 'FY','Q1','Q2','Q3','Q4'
    value          NUMERIC NOT NULL,
    unit           TEXT NOT NULL,               -- 'USD' | 'USD/share'
    start_date     DATE,
    end_date       DATE,
    UNIQUE (company_id, metric, fiscal_year, fiscal_period)
);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup
    ON financial_metrics (metric, fiscal_year, fiscal_period);

-- The RAG corpus: table schemas, column notes, and financial glossary entries, each with
-- a 1024-dim voyage-3.5-lite embedding. Retrieval runs a cosine (<=>) nearest-neighbour
-- search over this table BEFORE every SQL generation call.
CREATE TABLE IF NOT EXISTS rag_documents (
    id         SERIAL PRIMARY KEY,
    doc_type   TEXT NOT NULL,                   -- 'table_schema' | 'column' | 'glossary'
    title      TEXT NOT NULL UNIQUE,
    content    TEXT NOT NULL,
    embedding  VECTOR(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding
    ON rag_documents USING hnsw (embedding vector_cosine_ops);

-- Document-RAG layer: chunked narrative text from 10-K filings (Item 1A Risk
-- Factors and Item 7 MD&A), each with a voyage-3.5-lite embedding. A separate
-- retrieval path (app/filing_retrieval.py) searches this table for qualitative
-- questions, alongside the structured financial_metrics / text-to-SQL path.
-- Loaded by scripts/load_filings.py.
CREATE TABLE IF NOT EXISTS filing_chunks (
    id            SERIAL PRIMARY KEY,
    company_id    INT NOT NULL REFERENCES companies(id),
    accession     TEXT NOT NULL,          -- SEC accession number of the 10-K
    form          TEXT NOT NULL,          -- '10-K'
    fiscal_year   INT,                    -- report fiscal year label
    section       TEXT NOT NULL,          -- 'risk_factors' | 'mda'
    chunk_index   INT NOT NULL,           -- order within (company, section)
    content       TEXT NOT NULL,
    embedding     VECTOR(1024) NOT NULL,
    UNIQUE (company_id, accession, section, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_filing_chunks_embedding
    ON filing_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_filing_chunks_company
    ON filing_chunks (company_id, section);
