"""Shared pytest fixtures for the hermetic unit test suite.

WHY this file exists:
The unit tests must pass with NO network, NO database, and NO API keys —
that is the bar for a CI job that runs on every push. To make that true we:

1. Inject dummy environment variables *before* any `app.*` module is imported,
   because `app.config.Settings` (pydantic-settings) requires DATABASE_URL and
   would otherwise blow up at first `get_settings()` call.
2. Provide `patch_boundary`, a helper that monkeypatches a function at its
   *origin* module AND at any consumer module that imported it via
   `from x import y`. This is the classic "patch where it's looked up, not
   where it's defined" trap — we defensively patch both so the tests do not
   depend on the import style chosen inside app/pipeline.py or app/main.py.
3. Provide small factories for the pydantic models the pipeline shuttles
   around, so individual tests stay focused on behavior, not setup.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# --- Step 0: make `app` importable regardless of how pytest was invoked. ---
# pytest only puts the *test* directory on sys.path when there is no
# __init__.py; the repo root (which contains the `app` package) is what we
# actually need. Doing this in conftest.py guarantees it happens before any
# test module import.
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Step 1: dummy env vars BEFORE any app import. ---
# Load a real .env FIRST if one exists. WHY this matters: pytest imports every
# conftest.py during collection, so this module runs even when only the eval
# suite is selected — and because environment variables outrank .env in
# pydantic-settings, a dummy DATABASE_URL set here would silently hijack the
# eval run's real credentials and fail it with connection errors.
# Loading .env first makes the `setdefault` calls below do what they always
# claimed to: apply ONLY when nothing real is configured (the hermetic CI unit
# job), while a developer's .env or CI's real secrets win everywhere else.
load_dotenv(Path(ROOT) / ".env")

# The values below are syntactically valid but point nowhere — if any unit test
# accidentally opens a real connection it fails fast rather than silently
# reaching a live service.
os.environ.setdefault("DATABASE_URL", "postgresql://unit:unit@127.0.0.1:5/unittest")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key-not-real")
os.environ.setdefault("VOYAGE_API_KEY", "test-voyage-key-not-real")

import pytest  # noqa: E402  (env setup must precede app imports)

from app.models import ContextDoc, QueryResponse, SQLAttempt  # noqa: E402


@pytest.fixture
def patch_boundary(monkeypatch):
    """Patch `name` on its origin module and on every consumer module.

    Usage:
        patch_boundary("app.db", "execute_readonly", fake, consumers=("app.pipeline",))

    WHY: if app/pipeline.py did `from app.db import execute_readonly`, patching
    only `app.db.execute_readonly` would leave the pipeline holding the real
    function. We patch both sides so the tests are robust to either import
    style — we do not own those files and must not assume.
    """

    def _patch(origin: str, name: str, replacement, consumers: tuple[str, ...] = ()):
        origin_mod = importlib.import_module(origin)
        monkeypatch.setattr(origin_mod, name, replacement)
        for consumer in consumers:
            try:
                mod = importlib.import_module(consumer)
            except Exception:  # consumer not importable — nothing to patch
                continue
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, replacement)

    return _patch


@pytest.fixture
def sample_context_docs() -> list[ContextDoc]:
    """A tiny, realistic retrieval result.

    Mirrors what pgvector similarity search would return: schema + glossary
    docs sorted by descending similarity. Tests assert these flow through the
    pipeline untouched into QueryResponse.context_docs (transparency is a
    product feature here, so it gets tested).
    """
    return [
        ContextDoc(
            doc_type="table_schema",
            title="financial_metrics table",
            content=(
                "financial_metrics(company_id, filing_id, metric, fiscal_year, "
                "fiscal_period, value, unit) — one row per company/metric/period."
            ),
            similarity=0.91,
        ),
        ContextDoc(
            doc_type="column",
            title="metric column values",
            content=(
                "metric is one of: revenue, net_income, total_assets, "
                "total_liabilities, eps_diluted."
            ),
            similarity=0.87,
        ),
        ContextDoc(
            doc_type="glossary",
            title="net margin",
            content="net margin = net_income / revenue for the same fiscal period.",
            similarity=0.74,
        ),
    ]


@pytest.fixture
def make_attempt():
    """Factory for SQLAttempt with sensible defaults (override what matters)."""

    def _make(
        attempt_number: int = 1,
        sql: str = "SELECT 1",
        outcome: str = "success",
        error_message: str | None = None,
        correction_reasoning: str | None = None,
        duration_ms: int = 12,
    ) -> SQLAttempt:
        return SQLAttempt(
            attempt_number=attempt_number,
            sql=sql,
            outcome=outcome,
            error_message=error_message,
            correction_reasoning=correction_reasoning,
            duration_ms=duration_ms,
        )

    return _make


@pytest.fixture
def make_query_response(make_attempt):
    """Factory for a fully-populated successful QueryResponse.

    Used by the API tests to stand in for the real pipeline: the API layer's
    job is transport (validation, headers, serialization), so it gets a
    canned pipeline result and we assert only on transport concerns.
    """

    def _make(question: str, request_id: str, **overrides) -> QueryResponse:
        defaults: dict = {
            "request_id": request_id,
            "question": question,
            "success": True,
            "sql": "SELECT fm.value FROM financial_metrics fm",
            "rows": [{"value": 391035000000.0}],
            "answer": "Apple's fiscal 2023 revenue was about $391 billion.",
            "attempts": [make_attempt()],
            "context_docs": [],
        }
        defaults.update(overrides)
        return QueryResponse(**defaults)

    return _make
