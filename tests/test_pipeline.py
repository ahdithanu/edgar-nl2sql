"""Unit tests for app/pipeline.py — the agentic retry loop.

This is the showcase module, so the tests read like its spec:

- bad SQL then good SQL  -> exactly 2 attempts, outcomes recorded in order
- three failures         -> loop stops at MAX_ATTEMPTS=3, success=False,
                            readable failure narrative (no stack traces)
- empty result           -> retried with the contract's exact feedback string
- guard rejection        -> outcome guard_rejected, reason fed forward
- generation feedback    -> attempt N's prompt receives attempts 1..N-1
- never raises           -> even when every dependency explodes

Everything external is mocked at the app.pipeline module boundary (it uses
`from x import y` imports, so patching the names ON app.pipeline is what
actually intercepts the calls).
"""

from __future__ import annotations

import pytest

import app.pipeline as pipeline
from app.models import ContextDoc
from app.sql_guard import SQLGuardError

GOOD_SQL = "SELECT value FROM financial_metrics WHERE metric = 'revenue'"
BAD_SQL = "SELECT value FROM financial_metrix"  # misspelled table
ROWS = [{"value": 383285000000.0}]


@pytest.fixture(autouse=True)
def quiet_dependencies(monkeypatch, sample_context_docs):
    """Baseline happy-path mocks; individual tests override what they probe.

    autouse so no test can accidentally hit Voyage/Anthropic/Postgres — the
    unit suite must be hermetic.
    """
    monkeypatch.setattr(
        pipeline, "retrieve_context", lambda question: sample_context_docs
    )
    monkeypatch.setattr(
        pipeline, "generate_sql", lambda q, ctx, prior: (GOOD_SQL, None)
    )
    monkeypatch.setattr(pipeline, "validate_sql", lambda sql: sql)
    monkeypatch.setattr(pipeline, "execute_readonly", lambda sql: list(ROWS))
    monkeypatch.setattr(
        pipeline,
        "synthesize_answer",
        lambda q, sql, rows: "Apple's revenue was $383.3 billion.",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_success_on_first_attempt(sample_context_docs):
    resp = pipeline.run_pipeline("What was Apple's revenue?", "req-1")

    assert resp.success is True
    assert resp.sql == GOOD_SQL
    assert resp.rows == ROWS
    assert resp.answer == "Apple's revenue was $383.3 billion."
    assert resp.request_id == "req-1"
    assert len(resp.attempts) == 1
    assert resp.attempts[0].outcome == "success"
    assert resp.attempts[0].attempt_number == 1
    assert resp.attempts[0].correction_reasoning is None
    # Retrieval transparency: what was injected comes back verbatim.
    assert resp.context_docs == sample_context_docs


# ---------------------------------------------------------------------------
# The retry loop — the core contract
# ---------------------------------------------------------------------------


def test_bad_sql_then_good_sql_takes_two_attempts(monkeypatch):
    """THE canonical self-correction scenario from CONTRACTS.md."""
    calls: list[list] = []

    def fake_generate(question, context, prior_attempts):
        calls.append(list(prior_attempts))  # snapshot: what feedback did we get?
        if len(prior_attempts) == 0:
            return BAD_SQL, None
        return GOOD_SQL, "The table name was misspelled; corrected it."

    def fake_execute(sql):
        if sql == BAD_SQL:
            raise RuntimeError('relation "financial_metrix" does not exist')
        return list(ROWS)

    monkeypatch.setattr(pipeline, "generate_sql", fake_generate)
    monkeypatch.setattr(pipeline, "execute_readonly", fake_execute)

    resp = pipeline.run_pipeline("What was Apple's revenue?", "req-2")

    assert resp.success is True
    assert len(resp.attempts) == 2

    first, second = resp.attempts
    assert first.attempt_number == 1
    assert first.outcome == "execution_error"
    assert "financial_metrix" in (first.error_message or "")
    assert first.correction_reasoning is None

    assert second.attempt_number == 2
    assert second.outcome == "success"
    assert second.sql == GOOD_SQL
    assert second.correction_reasoning == (
        "The table name was misspelled; corrected it."
    )

    # The feedback channel: attempt 2's generate call saw attempt 1's failure.
    assert len(calls) == 2
    assert calls[0] == []
    assert len(calls[1]) == 1
    assert calls[1][0].outcome == "execution_error"


def test_cap_of_three_attempts_enforced(monkeypatch):
    generate_count = 0

    def always_bad(question, context, prior_attempts):
        nonlocal generate_count
        generate_count += 1
        return BAD_SQL, ("diagnosis" if prior_attempts else None)

    monkeypatch.setattr(pipeline, "generate_sql", always_bad)
    monkeypatch.setattr(
        pipeline,
        "execute_readonly",
        lambda sql: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    resp = pipeline.run_pipeline("Impossible question", "req-3")

    assert pipeline.MAX_ATTEMPTS == 3  # hardcoded by design — cost cap
    assert generate_count == 3
    assert resp.success is False
    assert resp.sql is None
    assert resp.rows == []
    assert len(resp.attempts) == 3
    assert [a.attempt_number for a in resp.attempts] == [1, 2, 3]
    assert all(a.outcome == "execution_error" for a in resp.attempts)


def test_failure_answer_is_readable_not_a_stack_trace(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "execute_readonly",
        lambda sql: (_ for _ in ()).throw(RuntimeError("syntax error at or near")),
    )

    resp = pipeline.run_pipeline("Unanswerable", "req-4")

    assert resp.success is False
    assert "Traceback" not in resp.answer
    assert "attempt" in resp.answer.lower()  # narrates what was tried
    assert resp.answer  # never empty, per QueryResponse contract


def test_empty_result_is_retried_with_contract_feedback(monkeypatch):
    executions: list[str] = []

    def fake_execute(sql):
        executions.append(sql)
        return [] if len(executions) == 1 else list(ROWS)

    monkeypatch.setattr(pipeline, "execute_readonly", fake_execute)

    resp = pipeline.run_pipeline("Q2 revenue?", "req-5")

    assert resp.success is True
    assert len(resp.attempts) == 2
    empty_attempt = resp.attempts[0]
    assert empty_attempt.outcome == "empty_result"
    # Exact wording from CONTRACTS.md — this string IS the retry prompt hint.
    assert empty_attempt.error_message == (
        "query returned no rows; the question likely has an answer — "
        "check metric names, fiscal_period values, joins"
    )


def test_guard_rejection_recorded_and_fed_back(monkeypatch):
    validations = 0

    def fake_validate(sql):
        nonlocal validations
        validations += 1
        if validations == 1:
            raise SQLGuardError("Statement contains a forbidden DELETE operation.")
        return sql

    monkeypatch.setattr(pipeline, "validate_sql", fake_validate)

    resp = pipeline.run_pipeline("Delete everything", "req-6")

    assert resp.success is True
    assert resp.attempts[0].outcome == "guard_rejected"
    assert "DELETE" in (resp.attempts[0].error_message or "")
    assert resp.attempts[1].outcome == "success"


# ---------------------------------------------------------------------------
# Never-raise guarantees
# ---------------------------------------------------------------------------


def test_generation_exception_never_raises(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "generate_sql",
        lambda q, c, p: (_ for _ in ()).throw(RuntimeError("anthropic 529")),
    )

    resp = pipeline.run_pipeline("Anything", "req-7")  # must not raise

    assert resp.success is False
    assert len(resp.attempts) == 3
    assert all("SQL generation failed" in (a.error_message or "") for a in resp.attempts)


def test_retrieval_failure_does_not_abort(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "retrieve_context",
        lambda q: (_ for _ in ()).throw(RuntimeError("voyage down")),
    )

    resp = pipeline.run_pipeline("Anything", "req-8")  # must not raise

    # Pipeline proceeds with empty context; here the mocked generator still
    # succeeds — the point is that a retrieval outage is not a 500.
    assert resp.success is True
    assert resp.context_docs == []


def test_synthesis_failure_does_not_flip_success(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "synthesize_answer",
        lambda q, s, r: (_ for _ in ()).throw(RuntimeError("synthesis down")),
    )

    resp = pipeline.run_pipeline("Anything", "req-9")

    # We have good SQL and real rows — a narration hiccup must not turn a
    # successful query into a failed request.
    assert resp.success is True
    assert resp.rows == ROWS
    assert resp.answer  # falls back to a canned pointer at rows/SQL


# ---------------------------------------------------------------------------
# Logging contract
# ---------------------------------------------------------------------------


def test_every_attempt_emits_sql_attempt_event(monkeypatch):
    events: list[dict] = []

    class FakeLogger:
        def info(self, event, **kwargs):
            if event == "sql_attempt":
                events.append(kwargs)

        def warning(self, event, **kwargs):
            pass

        def error(self, event, **kwargs):
            pass

    monkeypatch.setattr(pipeline, "logger", FakeLogger())

    executions: list[str] = []

    def fake_execute(sql):
        executions.append(sql)
        if len(executions) == 1:
            raise RuntimeError("bad column")
        return list(ROWS)

    monkeypatch.setattr(pipeline, "execute_readonly", fake_execute)

    pipeline.run_pipeline("Revenue?", "req-10")

    assert len(events) == 2  # one event per attempt, including the failure
    for event in events:
        # Exact field set from CONTRACTS.md.
        assert event["request_id"] == "req-10"
        for field in ("attempt_number", "sql", "outcome", "error_message", "duration_ms"):
            assert field in event
    assert events[0]["outcome"] == "execution_error"
    assert events[1]["outcome"] == "success"
