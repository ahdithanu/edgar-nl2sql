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

CI enforces `accuracy >= EVAL_MIN_ACCURACY`, set to **0.85** against a measured baseline
of **17/17 = 100%** (reproduced on two consecutive runs). Two deliberate choices there.
The floor sits *below* the baseline because the pipeline is LLM-backed and therefore
nondeterministic — a gate set at the baseline would fail on sampling noise and train
everyone to ignore it. And the gate is on **aggregate** accuracy rather than per item:
individual misses print full diagnostics but only the summary test fails the build, so one
unlucky roll cannot block a deploy while a genuine regression — which moves several items
at once — still trips it. The point: in an LLM system,
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

## Appendix A: a worked self-correction example

All three transcripts below are copied from real `sql_attempt` log events on the live
system (2026-07-19). Nothing is invented.

Once the corpus is properly loaded, the system answers essentially every golden question
on the first attempt — good for accuracy, inconvenient for demonstrating the retry loop.
So this example uses **fault injection**: the retrieved context was replaced with a doc
that describes a `net_margin` column which does not exist. Everything downstream — the
prompt, the guard, the database, the retry loop — is the ordinary production path. This
simulates the realistic failure where documentation has drifted from the actual schema.

**Question:** _"What was Apple's net margin in fiscal 2023?"_

**Attempt 1** (2,988 ms) — trusts the (bad) context and selects the column it was told about:

```sql
SELECT c.ticker, c.name, fm.fiscal_year, fm.fiscal_period, fm.net_margin
FROM financial_metrics fm
JOIN companies c ON c.id = fm.company_id
WHERE c.ticker = 'AAPL' AND fm.fiscal_year = 2023 AND fm.fiscal_period = 'FY'
```

Outcome `execution_error`. Postgres replies:

```
column fm.net_margin does not exist
LINE 6:     fm.net_margin
```

That error text is fed back verbatim. **This is the whole trick**: the database is a
better critic of SQL than any amount of prompt engineering, and its error messages are
already precise, actionable English.

**The model's diagnosis** (recorded as `correction_reasoning`, returned to the caller):

> The failure indicates that despite the context description mentioning a `net_margin`
> column, it does not actually exist in the table. This suggests `financial_metrics` is
> actually structured as a long/EAV format (metric name + value), where net margin must be
> derived by dividing the `net_income` metric value by the `revenue` metric value for the
> same company/fiscal period, rather than being stored as a precomputed column.

It inferred the real schema shape *from a single error message* and — notably — decided to
distrust its own context.

**Attempt 2** (5,837 ms) — self-joins the table to compute the ratio:

```sql
SELECT c.ticker, c.name, ni.fiscal_year, ni.fiscal_period,
       ni.value / rev.value AS net_margin
FROM financial_metrics ni
JOIN financial_metrics rev
  ON rev.company_id = ni.company_id
 AND rev.fiscal_year = ni.fiscal_year
 AND rev.fiscal_period = ni.fiscal_period
 AND rev.metric = 'revenue'
JOIN companies c ON c.id = ni.company_id
WHERE c.ticker = 'AAPL' AND ni.metric = 'net_income'
  AND ni.fiscal_year = 2023 AND ni.fiscal_period = 'FY'
```

Outcome `success`. **Final answer:** _"Apple's net margin for fiscal year 2023 was
approximately 25.3%, meaning net income represented about 25.3 cents of every dollar in
revenue."_ (Independently checked: $96,995M ÷ $383,285M = 25.31%.)

Total cost of recovery: one extra LLM call and about three seconds. The alternative —
returning `column fm.net_margin does not exist` to the user — would have been a bug report
instead of an answer.

## Appendix B: when the retry loop went wrong

The retry loop can also fail in an *interesting* way, and the eval harness is what caught
it. This is the strongest argument in this repo for why the eval is a first-class
deliverable rather than a checkbox.

The golden set includes a deliberately unanswerable question: _"What was the average
rainfall in Mordor during the Third Age?"_ Correct behavior is a graceful refusal. What
the logs actually showed:

1. **Attempt 1** — the model declined to write SQL and explained why. The pipeline had no
   category for "refusal", so it recorded a generic `execution_error` and retried.
2. **Attempt 2** — `SELECT NULL::text AS answer WHERE FALSE`. Zero rows, so the pipeline
   fed back its empty-result hint, which at the time read: _"query returned no rows; **the
   question likely has an answer** — check metric names, fiscal_period values, joins."_
3. **Attempt 3** — cornered by a false premise, the model produced valid SQL that returned
   exactly one row:

   ```sql
   SELECT 'This database contains only SEC EDGAR financial metrics ...
           it has no data on fictional locations like Mordor.' AS answer
   ```

   One row returned, so the pipeline recorded **`success=True`**.

The user-visible answer was honest — no hallucinated rainfall figures — which is exactly
why this is dangerous: the output looked fine while the machinery was quietly broken. The
system reported success for an unanswerable question, and the "SQL" was prose in a string
literal rather than a query. Generalize that pattern to a question that merely *looks*
answerable and you have a system that fabricates rows to satisfy its own retry loop.

Two root causes, both fixed:

- **No way to say "no."** The model's only exits were "produce SQL" or "fail", so refusal
  was misclassified as a retryable error. There is now an explicit protocol: the model
  replies `CANNOT_ANSWER: <reason>`, which raises `UnanswerableQuestionError` and
  terminates the loop immediately with `success=False` and the model's own reason. As a
  bonus this *saves* two LLM calls per unanswerable question.
- **Feedback that asserted a falsehood.** "The question likely has an answer" was a
  reasonable heuristic — empty results usually *are* filter typos — but stated as fact it
  pressured the model into manufacturing one. The hint now describes the likely causes
  without promising an answer exists, and explicitly names `CANNOT_ANSWER` as a legitimate
  way out.

Behavior after the fix, from the logs:

```
attempt_number=1  outcome=unanswerable
reason='The database contains SEC EDGAR financial metrics ... not weather or rainfall
        data, and has no concept of fictional locations or historical ages.'
success=False   attempts=1   rows=[]
```

One attempt, honest failure, two fewer API calls. The regression is pinned by
`tests/test_pipeline.py::test_unanswerable_question_short_circuits_without_retrying`.

**The lesson worth taking to an interview:** the bug was not in the SQL, the schema, or
the model. It was in the *shape of the feedback* the agent received. An agentic loop
optimizes for whatever signal you give it, so a loop that only rewards "produce a row"
will eventually produce a row by any means available. Give it a way to be right about
being unable to answer.

## Appendix C: proof that retrieval is load-bearing

An easy claim to make and an easy one to test. Replacing `retrieve_context` with a stub
that returns no documents, while leaving everything else untouched:

```
question: "Which company had the highest revenue in 2023?"
attempt_number=1  outcome=unanswerable
reason='No schema/context documents were provided describing the tables and columns
        ... so I cannot determine table/column names to construct a valid query.'
```

With retrieval, that same question is answered correctly on the first attempt. Without it,
the model does not hallucinate plausible table names — it correctly reports that it cannot
proceed. Retrieval is not a nice-to-have accuracy boost here; it is the only thing that
tells the system what database it is querying.
