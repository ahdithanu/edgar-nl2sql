"""The core NL→SQL pipeline: retrieve, then an explicit agentic retry loop.

============================================================================
This module is the showcase of the whole project. One request flows through:

    retrieve_context (ALWAYS FIRST — see app/retrieval.py for why)
        │
        ▼
    ┌─ attempt 1..MAX_ATTEMPTS ────────────────────────────────────┐
    │  generate_sql        Claude writes SQL (with prior failures  │
    │      │               + its own diagnosis on retries)         │
    │      ▼                                                       │
    │  validate_sql        static guard: single read-only SELECT,  │
    │      │               no DML/DDL, no catalog snooping         │
    │      ▼                                                       │
    │  execute_readonly    runs in a READ ONLY tx w/ timeout —     │
    │      │               defense in depth behind the guard       │
    │      ▼                                                       │
    │  classify outcome:                                           │
    │    success          → synthesize answer, return              │
    │    guard_rejected   → feed guard's reason back, retry        │
    │    execution_error  → feed DB error back, retry              │
    │    empty_result     → feed "no rows" hint back, retry        │
    └───────────────────────────────────────────────────────────────┘
        │ (all attempts exhausted)
        ▼
    honest failure: success=False, answer narrates what was tried

The loop is written as a plain, visible `for` loop on purpose — no
framework, no hidden recursion. The retry mechanism (error fed back into
the next prompt, model asked to diagnose before regenerating) is the
"agentic" part, and it should be readable in one screen.

Reliability contract: run_pipeline NEVER raises. Every LLM, guard, and DB
exception is caught, recorded as a failed attempt (or a failed request),
and reflected in the returned QueryResponse. The HTTP layer above this can
therefore always return a well-formed 200 with an honest payload.
============================================================================
"""

from __future__ import annotations

import time

from app.db import execute_readonly
from app.generation import UnanswerableQuestionError, generate_sql, synthesize_answer
from app.logging_config import get_logger
from app.models import ContextDoc, QueryResponse, SQLAttempt
from app.retrieval import retrieve_context
from app.sql_guard import SQLGuardError, validate_sql

logger = get_logger(__name__)

# Hardcoded by design — deliberately NOT a config setting. Each attempt is
# a paid Claude call (plus a DB round trip), so this constant is the hard
# cost cap per user request: worst case 3 generation calls + 1 synthesis
# call. Making it configurable invites someone to "fix" a failing question
# by cranking retries to 10, which multiplies cost and latency while hiding
# the real problem (bad context docs or a genuinely unanswerable question).
# Empirically, if the model can't fix its SQL given two rounds of concrete
# error feedback, a fourth attempt rarely helps.
MAX_ATTEMPTS = 3

# Hint fed back to the model when a query runs cleanly but returns nothing.
# An empty result is *ambiguous*: it might be the correct answer to a bad
# question, but far more often the SQL filtered on a value that doesn't
# exist (metric='netincome' instead of 'net_income', period='2023-Q4'
# instead of 'Q4'). We nudge the model toward the usual culprits.
# NOTE the careful wording: an earlier version asserted "the question likely
# has an answer", and the eval caught the consequence — for a genuinely
# unanswerable question that false premise pushed the model into fabricating a
# row (`SELECT '<prose>' AS answer`) to satisfy the loop. Feedback now points at
# the likely mistakes WITHOUT promising an answer exists, and names the refusal
# protocol as a legitimate way out.
_EMPTY_RESULT_FEEDBACK = (
    "query returned no rows; this usually means a filter didn't match stored "
    "values — check metric names, fiscal_period values, and joins. If the data "
    "genuinely isn't in this schema, reply CANNOT_ANSWER: <reason> instead. "
    "Never return explanatory prose inside a SELECT literal."
)


def _failure_narrative(question: str, attempts: list[SQLAttempt]) -> str:
    """Build the honest, human-readable answer for a fully failed request.

    No stack traces, no jargon dumps — a reviewer (or end user) should be
    able to read this and understand what was tried and where it got stuck.
    The full technical detail still lives in QueryResponse.attempts.
    """
    outcome_phrases = {
        "guard_rejected": "was rejected by the SQL safety guard",
        "execution_error": "failed during execution",
        "empty_result": "ran successfully but returned no rows",
    }
    lines = [
        f"I could not answer this question after {len(attempts)} attempt(s)."
    ]
    for attempt in attempts:
        phrase = outcome_phrases.get(attempt.outcome, attempt.outcome)
        detail = f" ({attempt.error_message})" if attempt.error_message else ""
        lines.append(f"Attempt {attempt.attempt_number} {phrase}{detail}.")
    lines.append(
        "The question may reference data outside this database (25 large-cap "
        "US companies, five core financial metrics, fiscal years 2020-2024), "
        "or may need rephrasing."
    )
    return " ".join(lines)


def run_pipeline(question: str, request_id: str) -> QueryResponse:
    """Answer a natural-language question with SQL over the EDGAR database.

    Never raises: all failures are folded into the QueryResponse.
    """
    # ------------------------------------------------------------------
    # STEP 1 — RETRIEVAL, BEFORE ANY GENERATION.
    # The retrieved docs ground everything that follows; without them the
    # model is guessing at our schema. If retrieval itself fails (Voyage
    # outage, DB down), we don't abort: generation with empty context will
    # almost certainly fail, but the retry loop will then produce an honest
    # failure narrative instead of a 500 — and the attempts log will show
    # exactly what happened.
    # ------------------------------------------------------------------
    context: list[ContextDoc] = []
    try:
        context = retrieve_context(question)
    except Exception as exc:  # noqa: BLE001 — pipeline must never raise
        logger.error(
            "retrieval_failed", request_id=request_id, error=str(exc)
        )

    attempts: list[SQLAttempt] = []

    # ------------------------------------------------------------------
    # STEP 2 — THE AGENTIC RETRY LOOP.
    # Plain for-loop, 1-based attempt numbers to match SQLAttempt's schema.
    # Each iteration: generate → guard → execute → classify. Any outcome
    # other than success appends a failed SQLAttempt whose error_message
    # becomes feedback in the NEXT attempt's prompt (see generation.py's
    # _format_prior_attempts) — that feedback loop is what lets the model
    # actually fix its mistakes rather than re-rolling blindly.
    # ------------------------------------------------------------------
    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()

        # --- Generate. correction_reasoning is the model's diagnosis of
        # the previous failure (None on attempt 1, by contract).
        try:
            sql, correction_reasoning = generate_sql(question, context, attempts)
        except UnanswerableQuestionError as exc:
            # The model concluded the schema cannot answer this question. That
            # is a CONCLUSION, not a failure to retry: burning two more LLM
            # calls won't conjure data that doesn't exist, and (as the eval
            # demonstrated) pressuring the model further tempts it to fake a
            # row. Stop here and report honestly.
            attempt = SQLAttempt(
                attempt_number=attempt_number,
                sql="",
                outcome="unanswerable",
                error_message=exc.reason,
                correction_reasoning=None,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            attempts.append(attempt)
            _log_attempt(request_id, attempt)
            logger.info("question_unanswerable", request_id=request_id, reason=exc.reason)
            return QueryResponse(
                request_id=request_id,
                question=question,
                success=False,
                sql=None,
                rows=[],
                answer=(
                    f"I can't answer that from this database. {exc.reason} "
                    "This dataset covers revenue, net income, total assets, total "
                    "liabilities, and diluted EPS for 25 large US public companies, "
                    "fiscal years 2020–2024."
                ),
                attempts=attempts,
                context_docs=context,
            )
        except Exception as exc:  # noqa: BLE001 — API errors, parse failures
            # Generation itself failed (Anthropic API error, or no SQL in
            # the response). Record it as an execution_error attempt — the
            # closest outcome in the fixed vocabulary — with a clearly
            # prefixed message so it can't be mistaken for a DB failure.
            # We keep looping: transient API errors often clear on retry.
            attempt = SQLAttempt(
                attempt_number=attempt_number,
                sql="",
                outcome="execution_error",
                error_message=f"SQL generation failed: {exc}",
                correction_reasoning=None,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            attempts.append(attempt)
            _log_attempt(request_id, attempt)
            continue

        # --- Guard. Static validation BEFORE the SQL touches the database:
        # exactly one read-only SELECT, no DML/DDL, no catalog snooping.
        # (Execution below is ALSO read-only at the transaction level —
        # defense in depth; the guard just fails faster and cheaper.)
        try:
            sql = validate_sql(sql)
        except SQLGuardError as exc:
            # Log the validation result explicitly (failed), separate from the
            # attempt event, so "did the guard pass?" is directly queryable in
            # the logs rather than inferred from the attempt outcome.
            reason = getattr(exc, "reason", None) or str(exc)
            logger.info(
                "sql_validated",
                request_id=request_id,
                attempt_number=attempt_number,
                passed=False,
                reason=reason,
            )
            attempt = SQLAttempt(
                attempt_number=attempt_number,
                sql=sql,
                outcome="guard_rejected",
                # The guard's reason is the feedback — e.g. "INSERT is not
                # allowed" tells the model exactly what to stop doing.
                error_message=reason,
                correction_reasoning=correction_reasoning,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            attempts.append(attempt)
            _log_attempt(request_id, attempt)
            continue

        # Validation passed. Logged as its own event so every request's log
        # trail shows the guard verdict on every attempt, pass or fail — not
        # only the rejections.
        logger.info(
            "sql_validated",
            request_id=request_id,
            attempt_number=attempt_number,
            passed=True,
        )

        # --- Execute inside a READ ONLY transaction with a statement
        # timeout and a row cap (see app/db.py).
        try:
            rows = execute_readonly(sql)
        except Exception as exc:  # noqa: BLE001 — any DB error is feedback
            attempt = SQLAttempt(
                attempt_number=attempt_number,
                sql=sql,
                outcome="execution_error",
                # Postgres error text ("column fm.reveune does not exist")
                # is exactly the feedback the model needs to self-correct.
                error_message=str(exc),
                correction_reasoning=correction_reasoning,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            attempts.append(attempt)
            _log_attempt(request_id, attempt)
            continue

        # --- Classify: zero rows is treated as a failure worth retrying,
        # not a success, because in this domain it almost always means a
        # filter mismatch rather than a genuinely empty answer.
        if not rows:
            attempt = SQLAttempt(
                attempt_number=attempt_number,
                sql=sql,
                outcome="empty_result",
                error_message=_EMPTY_RESULT_FEEDBACK,
                correction_reasoning=correction_reasoning,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            attempts.append(attempt)
            _log_attempt(request_id, attempt)
            continue

        # --- SUCCESS. Record the attempt, narrate the rows, return.
        attempt = SQLAttempt(
            attempt_number=attempt_number,
            sql=sql,
            outcome="success",
            error_message=None,
            correction_reasoning=correction_reasoning,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        attempts.append(attempt)
        _log_attempt(request_id, attempt)

        try:
            answer = synthesize_answer(question, sql, rows)
        except Exception as exc:  # noqa: BLE001 — never fail a good query
            # We have correct SQL and real rows — a synthesis hiccup must
            # not turn a successful query into a failed request. Fall back
            # to pointing at the data itself.
            logger.error(
                "synthesis_failed", request_id=request_id, error=str(exc)
            )
            answer = (
                f"The query succeeded and returned {len(rows)} row(s), but "
                f"the plain-English summary could not be generated. See the "
                f"rows and SQL in this response for the result."
            )

        return QueryResponse(
            request_id=request_id,
            question=question,
            success=True,
            sql=sql,
            rows=rows,
            answer=answer,
            attempts=attempts,
            context_docs=context,
        )

    # ------------------------------------------------------------------
    # STEP 3 — HONEST FAILURE. All attempts exhausted. No exception, no
    # stack trace: success=False and an answer that narrates what was
    # tried, so the caller (and the eval harness's graceful-failure case)
    # gets a complete, explainable response.
    # ------------------------------------------------------------------
    logger.warning(
        "pipeline_exhausted",
        request_id=request_id,
        question=question,
        attempt_count=len(attempts),
        outcomes=[a.outcome for a in attempts],
    )
    return QueryResponse(
        request_id=request_id,
        question=question,
        success=False,
        sql=None,
        rows=[],
        answer=_failure_narrative(question, attempts),
        attempts=attempts,
        context_docs=context,
    )


def _log_attempt(request_id: str, attempt: SQLAttempt) -> None:
    """Emit the per-attempt structured log event.

    One `sql_attempt` event per loop iteration, always with request_id, so
    a single log query reconstructs any request's full retry history —
    this is the operational story of the pipeline (grep request_id, read
    the attempts in order, see exactly where and why each one failed).
    """
    logger.info(
        "sql_attempt",
        request_id=request_id,
        attempt_number=attempt.attempt_number,
        sql=attempt.sql,
        outcome=attempt.outcome,
        error_message=attempt.error_message,
        duration_ms=attempt.duration_ms,
    )
