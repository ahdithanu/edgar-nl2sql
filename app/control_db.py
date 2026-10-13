"""Control-plane database access: the credential store and the audit log.

This is deliberately SEPARATE from app/db.py (the data plane that runs
LLM-generated SQL). The two planes have different trust levels and, in a
hardened deployment, different Postgres roles:

- Data plane  (app/db.py)      -> role `edgar_app`, runs model-written SELECTs
                                  under READ ONLY, over the public tables.
- Control plane (this module)  -> role `edgar_ctl`, reads app_meta.api_keys and
                                  APPENDS to app_meta.audit_log.

Keeping them apart means the very SQL we generate from user input can never
read a credential hash or tamper with an audit row: those tables live in the
`app_meta` schema, the SQL guard blocks that schema, and (when a dedicated
control role is configured) the data-plane role has no grant on it at all.

The pool here is small — auth lookups and audit appends are tiny, infrequent
queries compared to the data plane — and lazily created so unit tests and
module imports never require a database.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

_control_pool: ConnectionPool | None = None


def get_control_pool() -> ConnectionPool:
    """Return the process-wide control-plane pool, creating it on first use."""
    global _control_pool
    if _control_pool is None:
        _control_pool = ConnectionPool(
            conninfo=get_settings().control_dsn,
            min_size=1,
            max_size=3,  # auth + audit are light; keep the footprint small
            open=True,
        )
    return _control_pool


@contextmanager
def control_connection() -> Iterator:
    """A pooled control-plane connection with dict rows.

    Callers own the transaction: read paths should roll back, the audit writer
    commits. Kept as a thin context manager so auth.py and audit.py never touch
    the pool singleton directly (and so tests can monkeypatch this one seam).
    """
    with get_control_pool().connection() as conn:
        conn.row_factory = dict_row
        yield conn


def close_control_pool() -> None:
    """Close the control pool (FastAPI lifespan shutdown). Idempotent."""
    global _control_pool
    if _control_pool is not None:
        _control_pool.close()
        _control_pool = None
