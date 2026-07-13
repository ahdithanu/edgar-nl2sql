"""Postgres access layer: one lazy connection pool + a hardened read-only executor.

This module is the LAST line of defense against a bad or malicious generated query.
`app/sql_guard.py` statically rejects anything that isn't a single SELECT, but static
analysis can never be exhaustive — so execution ALSO runs inside a transaction that
Postgres itself enforces as READ ONLY, with a server-side statement timeout and a hard
cap on returned rows. Even if a hostile statement slipped past the guard, the database
would refuse to write, refuse to run forever, and refuse to flood us with rows.
"""

import datetime
import decimal
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

# Lazy singleton — created on first use, not at import time, so unit tests can import
# this module (and everything that imports it) without a database or DATABASE_URL.
_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Return the process-wide connection pool, creating it on first call.

    WHY a pool (even for a small demo): Supabase's pooler and Postgres both charge a
    real cost per connection setup; a `ConnectionPool` amortizes that and gives us
    health-checked, recycled connections for free.
    """
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=5,  # modest: Supabase pooler multiplexes behind this
            open=True,
        )
    return _pool


def _jsonable(value: Any) -> Any:
    """Coerce DB-native types to JSON-safe Python types.

    Postgres NUMERIC arrives as Decimal and DATE as datetime.date — neither survives
    `json.dumps`, and both would blow up FastAPI's response serialization and the LLM
    answer-synthesis step (which receives rows as plain dicts). Floats are fine for our
    domain: financial metrics at USD precision, compared with 1% tolerance in eval.
    """
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


def execute_readonly(sql: str) -> list[dict]:
    """Execute a (pre-validated) SELECT under strict read-only constraints.

    Layered constraints, all enforced server-side by Postgres:
      1. `SET TRANSACTION READ ONLY`  — any write attempt errors out, period.
      2. `SET LOCAL statement_timeout` — runaway queries (missing join predicate,
         accidental cross product) are killed instead of hogging the pool.
      3. A SERVER-SIDE cursor + `fetchmany(max_result_rows)` — bounds both the
         response size AND this process's memory, no matter what LIMIT the generated
         SQL did or didn't include. (A default client-side cursor would buffer the
         ENTIRE result set into worker memory on execute; a named cursor streams
         only the rows we actually fetch, so `SELECT * FROM generate_series(1, 1e9)`
         costs us at most max_result_rows rows of memory.)

    Raises on any database error — the pipeline catches it, records the attempt as
    `execution_error`, and feeds the message back to the model for self-correction.
    """
    settings = get_settings()
    with get_pool().connection() as conn:
        try:
            # psycopg opens the transaction implicitly on the first execute, so
            # this SET TRANSACTION is guaranteed to be its first statement.
            conn.execute("SET TRANSACTION READ ONLY")
            # SET does not accept bind parameters; the value is an int from our own
            # validated Settings, never user input, so an f-string is safe here.
            conn.execute(f"SET LOCAL statement_timeout = {settings.statement_timeout_ms}")
            # Named (server-side) cursor: rows stay on the server until fetched.
            with conn.cursor(name="readonly_result", row_factory=dict_row) as cur:
                cur.execute(sql)  # type: ignore[arg-type]  # LLM SQL is dynamic by nature
                raw_rows = cur.fetchmany(settings.max_result_rows)
            return [{key: _jsonable(val) for key, val in row.items()} for row in raw_rows]
        finally:
            # Nothing to persist in a read-only transaction; rolling back returns the
            # connection to the pool clean regardless of what happened above.
            conn.rollback()


def check_health() -> bool:
    """True if the database answers SELECT 1; False on ANY failure.

    Deliberately swallows all exceptions: /health must report 'degraded' rather than
    500 when the database is down — a monitoring endpoint that crashes when things are
    broken is useless.
    """
    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def close_pool() -> None:
    """Close the pool and forget it (FastAPI lifespan shutdown hook).

    Resetting the singleton to None lets tests cleanly re-create pools between cases.
    """
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
