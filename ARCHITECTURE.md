# Architecture

This document explains the design decisions behind edgar-nl2sql in plain English. It is
written for a smart reader who is not necessarily an LLM or database specialist — if you
know roughly what SQL and an API are, everything here should follow.

The one-sentence version: **a user asks a financial question in English; we look up the
most relevant schema documentation, hand it to Claude to write SQL, refuse to run
anything that isn't a read-only SELECT, and if the query fails we show Claude the error
and let it try again — at most three times.**

## The problem shape

"Text-to-SQL" sounds solved until you try it. Three things go wrong in practice:

1. **The model doesn't know your schema.** It will happily invent a `profits` table
   that doesn't exist.
2. **The model is occasionally wrong even when it knows the schema.** A misspelled
   column or a wrong join should not mean a broken user experience.
3. **You are handing an AI write-capable access to a database.** That should terrify
   you until the system makes it structurally impossible to do damage.

The three core mechanisms below — retrieval, the retry loop, and the SQL guard — map
one-to-one onto those three failure modes.

## Why retrieval happens *before* generation

The naive approach is to paste your entire schema into every prompt and hope. That works
for three tables — and this project only *has* three data tables — but it is the wrong
habit, because it stops working exactly when the project starts mattering (hundreds of
tables, thousands of columns, domain glossaries).

So the pipeline is retrieval-first by construction:

1. The user's question is turned into a 1024-dimensional vector (an *embedding*) using
   Voyage AI's `voyage-3.5-lite`. Think of an embedding as coordinates in "meaning
   space": questions about revenue land near documents about revenue.
2. We compare that vector against a pre-embedded corpus of ~30 short documents — table
   schemas, notes on tricky columns, and a financial glossary ("net margin =
   net_income / revenue, same period, which requires reading two rows") — using cosine
   similarity in pgvector.
3. The top-k most relevant documents are injected verbatim into Claude's prompt.

The model never answers from memory about our schema; it answers from documentation we
retrieved *for this specific question*. That has three concrete payoffs:

- **Grounding.** The glossary encodes house semantics no model could guess — e.g. that
  Q4 values for flow metrics are *derived* (FY minus Q1–Q3), or that balance-sheet
  metrics have no separate Q4 row. Retrieval is how domain knowledge gets into the
  prompt without fine-tuning anything.
- **Transparency.** The API returns `context_docs` with every response, so you can see
  exactly what the model was shown. When generation goes wrong, the first debugging
  question — "did it even have the right context?" — is answerable from the response.
- **Scalability of the pattern.** Swap the 30-document corpus for 3,000 documents and
  nothing else changes.

### Why the schema goes in the prompt at all

A reasonable question: why not let the model query `information_schema` and discover
the tables itself? Two reasons. First, we explicitly *block* system-catalog access (see
the guard below) — an LLM exploring your catalog is an information-disclosure surface.
Second, raw catalogs describe structure but not meaning. `fiscal_period TEXT` tells the
model nothing; our column doc explains that `'FY'` is a full year for flows but a
year-end snapshot for balances. Curated context beats discovered context.

## Why pgvector instead of a separate vector database

Because embeddings here are an *index over documentation*, not the product. A dedicated
vector database (Pinecone, Weaviate, Qdrant, …) makes sense when vector search is the
core workload at serious scale. Ours is ~30 documents queried once per request.

Keeping the vectors in the same Postgres that holds the financial data means:

- **One system to run, secure, back up, and connection-pool.** Every extra datastore
  is an extra failure mode, an extra set of credentials, and an extra thing to explain
  in an incident.
- **One query language.** The similarity search is a four-line SQL query
  (`ORDER BY embedding <=> $1 LIMIT k`) with an HNSW index. Nothing exotic.
- **No synchronization problem.** In a two-store design, the vector store and the
  source-of-truth database drift apart and you end up building a reconciliation
  pipeline. Here the corpus and its embeddings live in one transactional store.

At tens of millions of vectors with strict latency SLOs, this trade-off flips.
We are roughly six orders of magnitude below that.

## The self-correction loop (the showcase)

A single-shot NL→SQL system is only as good as the model's first draft. This pipeline
treats generation as *agentic*: attempt, observe, diagnose, retry.

```
retrieve context
└─ attempt 1..3:
   ├─ Claude generates SQL (on retries: shown every prior SQL + its error, asked to
   │  diagnose what went wrong BEFORE writing the new query)
   ├─ SQL guard validates              → rejected?  feed the reason back, retry
   ├─ execute (read-only, 10s timeout) → exception? feed the DB error back, retry
   ├─ zero rows                        → feed back "no rows — check metric names,
   │                                     fiscal_period values, joins", retry
   └─ rows returned → synthesize plain-English answer → done
```

Three distinct failure signals feed the loop, and they are deliberately different:

- **Guard rejection** — the model produced something we refuse to run (not a SELECT,
  multiple statements, catalog snooping). The rejection reason is precise, so the model
  usually fixes it immediately.
- **Execution error** — Postgres itself complained ("column c.tikcer does not exist").
  Database error messages are excellent few-shot feedback; models are very good at
  reading them.
- **Empty result** — the subtlest one. Zero rows is often not "there is no answer" but
  "you filtered on `metric = 'Revenue'` when the value is `'revenue'`". Treating empty
  results as a retryable signal (with a hint listing the usual suspects) catches a
  whole class of silent wrong-answers. The trade-off: for questions whose true answer
  is an empty set, we burn extra attempts — acceptable in this domain, where nearly
  every well-formed question has data behind it.

Every attempt is recorded in the response (`attempts[]`: the SQL, the outcome, the
error, and the model's own correction reasoning) and logged as a structured
`sql_attempt` event. The self-correction isn't a black box — it's the demo.

### Why the cap is exactly 3 — and hardcoded

Empirically, attempt 2 fixes most correctable errors (it has a concrete error message
to react to) and attempt 3 catches a useful remainder. Beyond that, the model is
usually stuck in a loop — rephrasing the same wrong idea — and each extra attempt costs
real money and multiplies worst-case latency (each attempt = an LLM call + a DB query).

`MAX_ATTEMPTS = 3` is a constant, not a config knob, on purpose: retry count is a
cost-and-latency policy, and policies that can drift via an env var eventually do
(someone sets 10 in production "to improve accuracy" and the p99 explodes). Changing
the policy should require a code change and a code review.

After three failures the API returns `success: false` with an honest, readable account
of what was tried. A clear "here's what I attempted and why it failed" beats a
confident hallucination every time.

## The SQL guard: never trust generated SQL

Generated SQL is untrusted input, full stop. Defense has two independent layers:

1. **Static validation** (`app/sql_guard.py`). The SQL is *parsed* with sqlglot — not
   regex-matched — and must be exactly one statement whose top-level node is a SELECT.
   Any DML/DDL anywhere in the tree (INSERT/UPDATE/DELETE/DROP/…), `SELECT INTO`, or a
   reference to `pg_catalog` / `information_schema` is rejected. Parsing matters
   because string matching is trivially bypassable (`/**/DROP`, nested statements);
   an AST is not fooled by formatting.
2. **Runtime enforcement** (`app/db.py`). Even SQL that passes the guard executes
   inside a transaction set to `READ ONLY` with a `statement_timeout` (10s) and a row
   cap (200). If the guard has a bug, Postgres itself refuses writes; if the model
   writes an accidental cartesian product, the timeout kills it.

Belt *and* suspenders, because each layer covers the other's blind spots: the guard
gives good error messages the retry loop can learn from; the read-only transaction is
the guarantee that doesn't depend on our parsing being perfect.

## Evals gate CI

The eval harness (`eval/`) is a golden set of ≥15 real questions. For each one we store
a **reference SQL query, not a hardcoded expected value** — the harness executes the
reference SQL for ground truth at eval time, runs the full pipeline on the English
question, and compares result sets (1% tolerance on numeric cells). This matters
because the data gets reloaded from EDGAR; pinned constants would rot, but reference
SQL stays true.

CI enforces `accuracy >= EVAL_MIN_ACCURACY` (0.75 to start; to be raised once the
baseline is measured — see the TODO in the README). The point: in an LLM system,
**correctness is a distribution, not a boolean**. Unit tests prove the plumbing
(the guard rejects DML, the retry loop caps at 3); only an executed eval can prove the
system still *answers questions correctly* after a prompt tweak, a model upgrade, or a
corpus edit. Wiring it into CI turns "did we get worse?" from a vibe into a blocking
check.

The two test tiers are strictly separated: unit tests are hermetic (LLM and DB mocked
at module boundaries, run on every PR in seconds, no secrets), while eval tests are
live and only run when `RUN_EVAL=1` and real credentials exist.

## Staging / production separation

`main` auto-deploys to a **staging** Railway service — but only after unit tests pass
and the eval gate clears. Production is a **manual promotion** of the same artifact
(steps in the README), and rollback is redeploying the previous known-good deployment.

Why a human in the loop for a portfolio project? Because eval accuracy ≥ 0.75 still
means some answers are wrong, and an LLM system's failure modes are qualitative in
ways a threshold doesn't capture (tone, verbosity, a subtly misleading answer that is
numerically "close enough"). Staging is where a human eyeballs a few real queries
before users see them. The app is stateless (all state in Supabase; schema changes
idempotent and additive), which is what makes redeploy-to-rollback safe.

## What we deliberately did NOT build

Scope discipline is a feature. Missing things below are decisions, not oversights:

- **No multi-cloud, no AWS/Terraform/Kubernetes.** One Railway service and one
  Supabase project deploy this system. Infrastructure should be proportionate to the
  workload; a k8s cluster here would be résumé-driven engineering.
- **No separate vector database** — covered above; pgvector in the existing Postgres.
- **No caching layer, no queue, no horizontal scaling.** At portfolio traffic, a
  connection pool and a stateless container are the right amount of scaling. The
  design leaves obvious seams (stateless app, pooled DB) for when that changes.
- **No fine-tuned model.** Retrieval + a curated glossary gets domain knowledge into
  the prompt for the cost of writing ~30 short documents. Fine-tuning is slower to
  iterate, costs more, and makes every schema change a training run.
- **No streaming responses / no chat memory.** Each request is one self-contained
  question → answer. Conversation state is a product decision that would double the
  surface area of the system without demonstrating anything new about NL→SQL.
- **No admin UI.** `context_docs` and `attempts` in the JSON response, plus structured
  logs, are the observability story. A dashboard would be decoration at this scale.

## Appendix: a worked self-correction example

<!-- TODO(post-deploy): Replace this block with a REAL example pulled from production
     structlog output after first deployment — the `sql_attempt` events for a request
     where attempt 1 failed (ideally an execution_error or empty_result) and attempt 2
     succeeded. Show: the question, attempt 1 SQL + error, the model's
     correction_reasoning, attempt 2 SQL, and the final answer. Real logs only —
     do not fabricate an example. -->

_To be added from real request logs after the first deployment._
