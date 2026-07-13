# edgar-nl2sql

Ask a financial question in plain English; get back the SQL that answers it, the rows it
returned, and a grounded natural-language answer — over real SEC EDGAR financial data.

**Live demo:** _coming soon_ <!-- TODO: replace with Railway production URL -->

```
"What was Apple's net margin in fiscal 2023?"
        │
        ▼
POST /query  →  { sql: "SELECT ...", rows: [...], answer: "Apple's net margin in
                  fiscal 2023 was about 25.3% (net income $97.0B / revenue $383.3B).",
                  attempts: [...], context_docs: [...] }
```

It is a small, complete **NL → SQL RAG system** with the operational parts that usually
get skipped in demos: a read-only SQL guard, a self-correcting retry loop, structured
JSON logging with request IDs, an executable golden-set eval that **gates CI**, and a
staged deploy pipeline. The design write-up lives in [ARCHITECTURE.md](ARCHITECTURE.md).

## How it works

1. **Retrieve** — the question is embedded (Voyage AI `voyage-3.5-lite`, 1024 dims) and
   matched via pgvector cosine similarity against a curated corpus of schema docs,
   column notes, and a financial glossary. Retrieval happens **before** generation:
   Claude only ever sees schema context that was fetched for this specific question.
2. **Generate** — Claude writes a single SELECT statement using the retrieved context.
3. **Guard** — the SQL is parsed with sqlglot and rejected unless it is exactly one
   read-only SELECT (no DML/DDL, no system-catalog snooping). Execution additionally
   runs inside a `READ ONLY` transaction with a statement timeout — defense in depth.
4. **Execute & self-correct** — on a guard rejection, execution error, or empty result,
   the error is fed back to Claude to diagnose and regenerate. Max **3 attempts**, then
   the API returns an honest failure explaining what was tried.
5. **Answer** — Claude summarizes the returned rows (and only the returned rows) into a
   short plain-English answer.

```mermaid
flowchart LR
    U([User question]) --> API[FastAPI<br/>POST /query]
    API --> R[Retrieval<br/>Voyage embed + pgvector<br/>cosine top-k]
    R --> G[Claude<br/>SQL generation]
    G --> V[SQL guard<br/>sqlglot: SELECT-only]
    V --> X[(Supabase Postgres<br/>READ ONLY txn)]
    X -->|rows| A[Claude<br/>answer synthesis]
    A --> RESP([SQL + rows + answer<br/>+ attempts + context])
    V -.->|rejected| RETRY{{Retry loop<br/>max 3 attempts}}
    X -.->|error / 0 rows| RETRY
    RETRY -.->|error fed back<br/>for diagnosis| G
    subgraph Supabase Postgres + pgvector
        X
        E[(rag_documents<br/>1024-dim embeddings)]
    end
    R --- E
```

## Stack

| Concern | Choice |
|---|---|
| Data + vectors | **Supabase Postgres with pgvector** — relational data and embeddings in one database |
| Embeddings | **Voyage AI `voyage-3.5-lite`** (1024 dimensions) |
| SQL generation & answers | **Claude** (`anthropic` SDK, model configurable via `CLAUDE_MODEL`) |
| API | FastAPI + uvicorn, Python 3.12 |
| SQL validation | sqlglot (Postgres dialect) |
| Logging | structlog, JSON to stdout, request-ID bound |
| Data source | SEC EDGAR XBRL company facts (25 large-cap companies, FY2020–2024) |

## Setup

Prereqs: Python 3.12, a Supabase project (pgvector is available out of the box),
an Anthropic API key, a Voyage AI API key.

```bash
git clone <this-repo> && cd edgar-nl2sql
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env        # then fill in DATABASE_URL and the API keys

# 1. Create tables + pgvector extension (idempotent)
psql "$DATABASE_URL" -f scripts/schema.sql

# 2. Load SEC EDGAR financials for the 25 tracked companies (idempotent upserts)
python scripts/load_edgar.py

# 3. Embed the schema/glossary corpus into rag_documents
python scripts/build_embeddings.py

# 4. Run the API
uvicorn app.main:app --reload
```

Then:

```bash
curl -s localhost:8000/health

curl -s localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "What was Apple'\''s revenue in fiscal 2023?"}' | jq
```

## Example queries

- "What was Apple's revenue in fiscal 2023?"
- "Which company had the highest net income in 2024?"
- "What was Microsoft's net margin in fiscal 2023?"
- "Compare Amazon and Walmart revenue for fiscal 2022."
- "How did Nvidia's revenue grow year over year from 2022 to 2024?"
- "What was Tesla's debt ratio at the end of fiscal 2023?"
- "What was Coca-Cola's diluted EPS in Q2 of fiscal 2024?"

Every response includes the `attempts` array (each generated SQL, its outcome, and the
model's correction reasoning on retries) and `context_docs` (what retrieval injected) —
the pipeline shows its work.

## Tests & eval

```bash
pytest -m "not eval"        # unit tests: hermetic, no DB or API keys needed

RUN_EVAL=1 pytest -m eval   # golden-set eval: needs live DB + API keys
```

The eval harness executes each golden question's `reference_sql` for ground truth, runs
the full pipeline (retry loop included), and compares result sets with a 1% numeric
tolerance. CI fails if accuracy drops below `EVAL_MIN_ACCURACY` (default 0.75).

**Current golden-set accuracy:** _TBD — will be published after the first baseline run
against the live database._

## Docker

```bash
docker build -t edgar-nl2sql .

docker run --rm -p 8000:8000 --env-file .env edgar-nl2sql
# or with an explicit port:
docker run --rm -e PORT=8080 -p 8080:8080 --env-file .env edgar-nl2sql
```

The image is multi-stage (`python:3.12-slim` runtime, dependencies baked in a builder
stage) and runs as a non-root user.

## Deployment (Railway)

CI (`.github/workflows/ci.yml`) is a trust ladder:

1. **unit** — every push/PR, no secrets required.
2. **eval** — runs when `DATABASE_URL`, `ANTHROPIC_API_KEY`, and `VOYAGE_API_KEY`
   secrets are configured; gates on golden-set accuracy.
3. **deploy-staging** — on push to `main`, only after unit + eval are green, deploys to
   the Railway **staging** service via `railway up` (requires the `RAILWAY_TOKEN`
   secret; skipped gracefully when absent).

### Staging → production promotion

Production deploys are deliberately manual — a human looks at staging first:

1. CI deploys `main` to the `edgar-nl2sql-staging` service automatically.
2. Smoke-check staging: `GET /health` returns `ok`, and a couple of the example
   queries above return sensible SQL + answers.
3. Promote the same commit to production:
   ```bash
   railway up --service edgar-nl2sql-production --detach
   ```
   (run locally with `RAILWAY_TOKEN` set, or redeploy the staging image from the
   Railway dashboard onto the production service).

### Rollback

Railway keeps previous deployments per service:

1. Open the production service in the Railway dashboard → **Deployments**.
2. Pick the last known-good deployment → **Redeploy**. This reverts the running code
   without touching the database (the app is stateless; schema changes are additive
   and idempotent by design).
3. Equivalent CLI: `railway redeploy --service edgar-nl2sql-production` after checking
   out the known-good commit, or `railway down` to stop a bad deploy immediately.

## Repo map

```
app/            FastAPI service: config, models, db, retrieval, generation,
                sql_guard, pipeline (the retry loop lives here), main
scripts/        schema.sql, load_edgar.py (SEC EDGAR loader),
                context_docs.py + build_embeddings.py (RAG corpus)
tests/          hermetic unit tests (mocked LLM/DB)
eval/           golden_set.yaml + eval harness (live, accuracy-gated)
ARCHITECTURE.md design decisions and trade-offs, in plain English
```
