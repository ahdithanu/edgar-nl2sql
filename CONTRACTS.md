# CONTRACTS — edgar-nl2sql

Single source of truth for the multi-agent build. Every module MUST code against these
signatures exactly. If you believe a contract is wrong, implement it anyway and flag the
concern in your final report — do NOT unilaterally change another agent's files.

## Project

Natural-language → SQL RAG system over SEC EDGAR financial data. Portfolio project for a
Forward Deployed Engineer job search: correctness, readability, eval harness, and
operational maturity are the point. A user asks a financial question in English; we
retrieve schema/glossary context via pgvector similarity search, generate SQL with Claude,
validate it read-only, execute against Postgres, self-correct on failure (max 3 attempts),
and return SQL + rows + a plain-English answer.

## Stack (fixed, do not substitute)

- Python 3.12, FastAPI, uvicorn
- Supabase Postgres + pgvector (relational data AND embeddings in one DB)
- `psycopg[binary]` + `psycopg_pool` (psycopg 3, sync; FastAPI runs sync endpoints in its threadpool)
- `anthropic` SDK for SQL generation + answer synthesis (model from env, default `claude-sonnet-5`)
- `voyageai` for embeddings: model `voyage-3.5-lite`, 1024 dims (`input_type="document"` for corpus, `"query"` for questions)
- `sqlglot` for SQL validation parsing
- `structlog` for JSON structured logging
- `pydantic` v2 + `pydantic-settings`
- `httpx` for EDGAR calls, `pyyaml` for the golden set
- `pytest` for tests/eval

## File ownership (one owner per file — never touch files you don't own)

| Agent | Files |
|---|---|
| foundation | `pyproject.toml`, `app/__init__.py`, `app/config.py`, `app/models.py`, `app/logging_config.py`, `app/db.py`, `scripts/schema.sql`, `.env.example`, `.gitignore` |
| data | `scripts/load_edgar.py`, `scripts/context_docs.py`, `scripts/build_embeddings.py` |
| rag-core | `app/retrieval.py`, `app/generation.py`, `app/pipeline.py` |
| api | `app/sql_guard.py`, `app/main.py` |
| tests-eval | `tests/conftest.py`, `tests/test_sql_guard.py`, `tests/test_pipeline.py`, `tests/test_api.py`, `eval/golden_set.yaml`, `eval/test_eval.py` |
| devops-docs | `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`, `README.md`, `ARCHITECTURE.md` |

## Database schema (`scripts/schema.sql` — idempotent, exactly this)

```sql
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

CREATE TABLE IF NOT EXISTS rag_documents (
    id         SERIAL PRIMARY KEY,
    doc_type   TEXT NOT NULL,                   -- 'table_schema' | 'column' | 'glossary'
    title      TEXT NOT NULL UNIQUE,
    content    TEXT NOT NULL,
    embedding  VECTOR(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding
    ON rag_documents USING hnsw (embedding vector_cosine_ops);
```

Data semantics (bake these into glossary docs + eval):
- Flow metrics (`revenue`, `net_income`, `eps_diluted`): rows for `FY` and `Q1`–`Q4`.
  `Q4` is DERIVED as `FY − (Q1+Q2+Q3)` (approximate for EPS; documented caveat). `FY` ≠ sum-of-quarters queries should prefer `fiscal_period = 'FY'`.
- Instant/balance metrics (`total_assets`, `total_liabilities`): rows for `Q1`–`Q3` and `FY` (FY row = fiscal-year-end snapshot; there is no separate Q4 row).
- `fiscal_year` is the COMPANY's fiscal year label (e.g. AAPL FY2023 ends Sep 2023).

## Module contracts

### `app/config.py`
```python
class Settings(BaseSettings):          # pydantic-settings, reads .env
    database_url: str                  # Supabase pooler URL
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    claude_model: str = "claude-sonnet-5"
    embed_model: str = "voyage-3.5-lite"
    retrieval_top_k: int = 8
    max_result_rows: int = 200
    statement_timeout_ms: int = 10000
    log_level: str = "INFO"

@lru_cache
def get_settings() -> Settings: ...
```

### `app/models.py` (pydantic v2)
```python
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

class ContextDoc(BaseModel):
    doc_type: str
    title: str
    content: str
    similarity: float

class SQLAttempt(BaseModel):
    attempt_number: int                       # 1-based
    sql: str
    outcome: Literal["success", "guard_rejected", "execution_error", "empty_result"]
    error_message: str | None = None
    correction_reasoning: str | None = None   # model's diagnosis that produced THIS attempt (None on attempt 1)
    duration_ms: int

class QueryResponse(BaseModel):
    request_id: str
    question: str
    success: bool
    sql: str | None                           # final successful SQL, None if all attempts failed
    rows: list[dict] = []
    answer: str                               # plain-English answer OR clear failure explanation of what was tried
    attempts: list[SQLAttempt]
    context_docs: list[ContextDoc] = []       # what retrieval injected (transparency/demo value)

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    version: str = "0.1.0"
```

### `app/logging_config.py`
```python
def configure_logging(level: str = "INFO") -> None   # structlog JSON to stdout
def get_logger(name: str) -> structlog.BoundLogger
```
Every log event during a request is bound with `request_id`. Pipeline logs one event per
attempt: `event="sql_attempt"`, fields: request_id, attempt_number, sql, outcome,
error_message, duration_ms.

### `app/db.py`
```python
def get_pool() -> ConnectionPool                      # lazy singleton from settings.database_url
def execute_readonly(sql: str) -> list[dict]
    # Runs inside a transaction with: SET TRANSACTION READ ONLY;
    # SET LOCAL statement_timeout = settings.statement_timeout_ms;
    # fetches at most settings.max_result_rows rows; returns list of dicts
    # (column name -> value; Decimal -> float, date -> isoformat str for JSON safety).
def check_health() -> bool                            # SELECT 1, False on any exception
def close_pool() -> None
```

### `app/sql_guard.py`
```python
class SQLGuardError(Exception): ...                   # .reason: str
def validate_sql(sql: str) -> str                     # returns normalized SQL or raises
```
Rules (parse with sqlglot, dialect "postgres"): exactly one statement; top-level node must
be SELECT (CTEs/WITH allowed only if final expression is SELECT); reject any DML/DDL node
anywhere (INSERT/UPDATE/DELETE/MERGE/DROP/ALTER/CREATE/TRUNCATE/GRANT/REVOKE/COPY/CALL/SET/
EXECUTE); reject `INTO`; reject statements that fail to parse; reject access to
`pg_catalog`/`information_schema`/`pg_` tables (schema snooping); strip trailing semicolon.
Defense in depth: execution ALSO runs in a READ ONLY transaction (db.py).

### `app/retrieval.py`  — RETRIEVAL HAPPENS BEFORE GENERATION (comment this explicitly)
```python
def embed_query(text: str) -> list[float]             # voyage, input_type="query"
def retrieve_context(question: str, k: int | None = None) -> list[ContextDoc]
    # k defaults to settings.retrieval_top_k; pgvector cosine: ORDER BY embedding <=> %s::vector
    # similarity = 1 - cosine_distance
```

### `app/generation.py`
```python
def generate_sql(question: str, context: list[ContextDoc],
                 prior_attempts: list[SQLAttempt]) -> tuple[str, str | None]
    # returns (sql, correction_reasoning). correction_reasoning None when prior_attempts empty.
    # Prompt: retrieved context injected verbatim; on retry, include each prior attempt's SQL +
    # outcome + error and ask Claude to diagnose before regenerating. Ask for SQL in a
    # ```sql fenced block; parse it out defensively.
def synthesize_answer(question: str, sql: str, rows: list[dict]) -> str
    # Short plain-English answer grounded ONLY in rows. If rows truncated at max_result_rows, say so.
```

### `app/pipeline.py` — THE CORE. Explicit agentic retry loop.
```python
MAX_ATTEMPTS = 3   # hardcoded by design: caps LLM cost; do NOT make configurable

def run_pipeline(question: str, request_id: str) -> QueryResponse
```
Flow: retrieve_context → loop attempt 1..3: generate_sql → validate_sql (SQLGuardError ⇒
outcome guard_rejected, feed reason back as the error) → execute_readonly (exception ⇒
execution_error) → 0 rows ⇒ empty_result (feed back "query returned no rows; the question
likely has an answer — check metric names, fiscal_period values, joins") → rows ⇒ success,
synthesize_answer, return. Every attempt appended to attempts[] and logged via
`sql_attempt` event. After 3 failures: success=False, answer = readable summary of what
was tried and why it failed (no stack traces). All LLM/DB exceptions caught — the pipeline
NEVER raises to the caller.

### `app/main.py`
- `POST /query` → QueryResponse (generates `request_id` uuid4 hex, binds to structlog contextvars; also returned in `X-Request-ID` header)
- `GET /health` → HealthResponse (checks DB; "degraded" if DB down, still HTTP 200)
- Global exception handler: JSON `{"detail": ..., "request_id": ...}`, never a stack trace. 422 on validation errors.
- Lifespan: configure_logging on startup, close_pool on shutdown.

## Data loader (`scripts/load_edgar.py`)

Repeatable, idempotent (`ON CONFLICT` upserts). CLI: `python scripts/load_edgar.py [--tickers AAPL,MSFT] [--dry-run]`.

- Companies (25): AAPL MSFT GOOGL AMZN NVDA META TSLA JPM BAC GS V MA WMT COST HD KO PEP MCD XOM CVX JNJ PFE UNH DIS NFLX
- CIK mapping: `https://www.sec.gov/files/company_tickers.json`
- Facts: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json`
- Industry: `https://data.sec.gov/submissions/CIK{cik:0>10}.json` → `sicDescription`
- Headers: `User-Agent: edgar-nl2sql portfolio project ahdi@uaconsulting.co` on EVERY request; ≤10 req/s (sleep 0.15s between calls); retry 429/5xx with backoff.
- us-gaap tag fallbacks (first tag present wins, per metric):
  - revenue: `RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet`
  - net_income: `NetIncomeLoss`
  - total_assets: `Assets`
  - total_liabilities: `Liabilities`
  - eps_diluted: `EarningsPerShareDiluted`
- Units: USD for flows/balances (`units.USD`), `USD/shares` for EPS (store unit `USD/share`).
- Period selection (EDGAR facts include prior-year comparatives with the FILING's fy/fp —
  dedupe carefully): for each (fy, fp) group take the entry with the LATEST `end` date
  (that's the current period, not a comparative); annual flows = form 10-K, fp FY, duration
  330–380 days; quarterly flows = form 10-Q, fp Q1–Q3, duration 70–100 days; instants have
  no duration — take max `end` per (fy, fp). Keep fiscal years 2020–2024.
- Derive Q4 flows = FY − (Q1+Q2+Q3) when all three quarters present; attach Q4 row to the
  10-K filing; skip and log a warning when quarters are missing.
- Insert filings from each kept fact's `accn`/`form`/`filed`.
- Print a load summary table (company × metric row counts) at the end; nonzero exit on hard failure.

## Context corpus (`scripts/context_docs.py` + `build_embeddings.py`)

`context_docs.py` exports `CONTEXT_DOCS: list[dict]` with keys `doc_type,title,content` —
the RAG corpus, ~25–35 docs:
- `table_schema` (3): one per table — full DDL + row semantics + join keys.
- `column` (~8): tricky columns — `metric` (list the 5 exact values), `fiscal_period`
  (FY vs quarters, flow vs instant semantics, derived Q4 caveat), `fiscal_year`, `unit`, `ticker` vs `name`, etc.
- `glossary` (~15–20): net margin = net_income/revenue (same period, from two rows — needs self-join or FILTER),
  YoY growth, debt ratio = total_liabilities/total_assets, equity ≈ assets − liabilities, EPS,
  "latest year available is fiscal 2024", "biggest/largest by revenue means ORDER BY ... DESC LIMIT",
  quarter-over-quarter, TTM caveat, example SQL patterns for ratio and comparison questions.

`build_embeddings.py`: embeds all docs (voyage `input_type="document"`, batches of 128),
upserts into `rag_documents` on `title` (delete-and-reinsert on rerun is fine), prints count.

## Eval harness (`eval/`)

`golden_set.yaml`: ≥15 items:
```yaml
- id: revenue_apple_2023
  question: "What was Apple's revenue in fiscal 2023?"
  reference_sql: "SELECT fm.value FROM financial_metrics fm JOIN companies c ON ..."
  check: relative      # relative (numeric cells, 1% tolerance) | exact (sorted row sets)
```
Expected values are NOT hardcoded — the harness executes `reference_sql` for ground truth,
runs the full pipeline (retry loop included) on `question`, and compares result sets:
normalize rows (sorted, numeric coercion), `relative` allows 1% per numeric cell.
Cover: simple lookups, superlatives, ratios (net margin), YoY growth, multi-company
comparison, quarterly lookups, a nonsense question (expects graceful failure = pass when
`success=False` and answer explains), ambiguous phrasing.
`eval/test_eval.py`: pytest, `@pytest.mark.eval`, one test per item + a summary test that
prints an accuracy table and asserts `accuracy >= float(os.environ.get("EVAL_MIN_ACCURACY", "0.75"))`.
Eval tests auto-skip unless `RUN_EVAL=1` (they need live DB + API keys).
Unit tests (`tests/`) MUST run with no DB/network: mock anthropic/voyage/psycopg at module
boundaries. `tests/test_pipeline.py` proves the retry loop: mock generation returning bad
SQL then good SQL; assert 2 attempts logged, outcomes correct, cap of 3 enforced.

## DevOps

- `Dockerfile`: multi-stage (builder installs into venv; slim `python:3.12-slim` runtime,
  non-root user, `COPY app/ scripts/`), `CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`.
- `.github/workflows/ci.yml`: job `unit` (always: ruff not required, pytest -m "not eval");
  job `eval` (only when secrets available: needs DATABASE_URL, ANTHROPIC_API_KEY,
  VOYAGE_API_KEY; RUN_EVAL=1; gate on EVAL_MIN_ACCURACY); job `deploy-staging` (on push to
  main, needs unit+eval: `railway up` via RAILWAY_TOKEN — guard with `if: secrets present`,
  document manual production promotion in README).
- `.env.example`: DATABASE_URL, ANTHROPIC_API_KEY, VOYAGE_API_KEY, CLAUDE_MODEL, LOG_LEVEL — with comments, no real values.

## Style rules (all agents)

- Type hints everywhere; docstrings explain WHY (this repo is read by hiring managers).
- Comment the two showcase flows heavily: retrieval-before-generation and the retry loop.
- No secrets anywhere in the repo. No `print` in app code (structlog only); scripts may print.
- Keep it small and correct over clever. Python 3.12 syntax (e.g. `str | None`).
