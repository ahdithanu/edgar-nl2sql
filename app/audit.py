"""Tamper-evident audit log for every /query.

Two properties make this an audit trail rather than just another log:

1. Append-only at the DATABASE level. The application role is granted INSERT on
   app_meta.audit_log but NOT UPDATE or DELETE (see scripts/002_auth_audit.sql).
   The app literally cannot alter or erase a past entry.

2. Hash-chained for tamper-evidence. Each row stores a SHA-256 `entry_hash`
   computed over a canonical serialization of its own fields PLUS the previous
   row's `entry_hash`. Editing, reordering, or deleting any row (by someone who
   bypasses the grant, e.g. a DBA) breaks the chain from that point on, and
   verify_chain() will find the exact row where it broke.

What is recorded: who asked (principal), from where (client_ip), the natural-
language question, the generated SQL, the outcome, and a SHA-256 of the returned
rows (`result_hash`) plus the row count — proving WHAT was returned without
copying potentially large or sensitive result sets into the audit table.

write_audit is best-effort: it must NEVER turn a served answer into an error, so
every failure is logged and swallowed. Availability of the answer path does not
depend on the audit path succeeding.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field

from app.control_db import control_connection
from app.logging_config import get_logger

logger = get_logger(__name__)

# The genesis link for the very first row's prev_hash. Any fixed constant works;
# it just has to be stable so the chain is reproducible from the start.
GENESIS_HASH = "GENESIS"

# A single advisory-lock key serializes concurrent audit appends so two requests
# can't read the same prev_hash and fork the chain. Arbitrary but fixed.
_AUDIT_LOCK_KEY = 0x4544474152_00  # "EDGAR" + tag, fits in bigint


@dataclass
class AuditRecord:
    """Everything captured about one answered (or refused) request."""

    request_id: str
    principal_label: str
    question: str
    success: bool
    occurred_at: dt.datetime = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )
    principal_id: str | None = None
    client_ip: str | None = None
    mode: str | None = None
    sql: str | None = None
    attempt_count: int = 0
    outcomes: list[str] = field(default_factory=list)
    row_count: int | None = None
    result_hash: str | None = None
    duration_ms: int | None = None


def result_hash(rows: list[dict]) -> str:
    """SHA-256 over the returned rows — proves content without storing it."""
    payload = json.dumps(rows, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_payload(record: AuditRecord) -> str:
    """Deterministic serialization of the fields the hash commits to.

    Field set and ordering are fixed here on purpose: the chain is only
    verifiable if the bytes hashed at write time can be reproduced exactly at
    verify time, so this must stay stable (append new fields at the end).
    """
    fields = {
        "request_id": record.request_id,
        "occurred_at": record.occurred_at.astimezone(dt.timezone.utc).isoformat(),
        "principal_id": record.principal_id,
        "principal_label": record.principal_label,
        "client_ip": record.client_ip,
        "question": record.question,
        "mode": record.mode,
        "success": record.success,
        "sql": record.sql,
        "attempt_count": record.attempt_count,
        "outcomes": record.outcomes,
        "row_count": record.row_count,
        "result_hash": record.result_hash,
        "duration_ms": record.duration_ms,
    }
    return json.dumps(fields, sort_keys=True, default=str, separators=(",", ":"))


def compute_entry_hash(prev_hash: str, canonical: str) -> str:
    """entry_hash = SHA-256(prev_hash || '\\n' || canonical_payload)."""
    return hashlib.sha256(f"{prev_hash}\n{canonical}".encode("utf-8")).hexdigest()


def write_audit(record: AuditRecord) -> None:
    """Append one hash-chained row. Best-effort — never raises to the caller."""
    try:
        with control_connection() as conn:
            # Serialize appenders so the prev_hash read + insert is atomic
            # against other requests (released at transaction end).
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_AUDIT_LOCK_KEY,))
            last = conn.execute(
                "SELECT entry_hash FROM app_meta.audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = last["entry_hash"] if last else GENESIS_HASH

            canonical = canonical_payload(record)
            entry_hash = compute_entry_hash(prev_hash, canonical)

            conn.execute(
                """
                INSERT INTO app_meta.audit_log (
                    request_id, occurred_at, principal_id, principal_label,
                    client_ip, question, mode, success, sql, attempt_count,
                    outcomes, row_count, result_hash, duration_ms,
                    prev_hash, entry_hash
                ) VALUES (
                    %(request_id)s, %(occurred_at)s, %(principal_id)s,
                    %(principal_label)s, %(client_ip)s, %(question)s, %(mode)s,
                    %(success)s, %(sql)s, %(attempt_count)s, %(outcomes)s,
                    %(row_count)s, %(result_hash)s, %(duration_ms)s,
                    %(prev_hash)s, %(entry_hash)s
                )
                """,
                {
                    "request_id": record.request_id,
                    "occurred_at": record.occurred_at,
                    "principal_id": record.principal_id,
                    "principal_label": record.principal_label,
                    "client_ip": record.client_ip,
                    "question": record.question,
                    "mode": record.mode,
                    "success": record.success,
                    "sql": record.sql,
                    "attempt_count": record.attempt_count,
                    "outcomes": record.outcomes,
                    "row_count": record.row_count,
                    "result_hash": record.result_hash,
                    "duration_ms": record.duration_ms,
                    "prev_hash": prev_hash,
                    "entry_hash": entry_hash,
                },
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 — audit must never break the request
        logger.error("audit_write_failed", request_id=record.request_id, error=str(exc))


def verify_chain(rows: list[dict]) -> int | None:
    """Verify a hash chain. Returns the id of the first broken row, or None.

    `rows` are audit_log rows (dicts) ordered by ascending id, each containing
    the persisted fields plus prev_hash and entry_hash. Recomputes each link and
    checks it against what was stored and against the previous row's entry_hash.
    None means the chain is intact.
    """
    prev = GENESIS_HASH
    for row in rows:
        record = AuditRecord(
            request_id=row["request_id"],
            occurred_at=row["occurred_at"],
            principal_id=str(row["principal_id"]) if row.get("principal_id") else None,
            principal_label=row["principal_label"],
            client_ip=row.get("client_ip"),
            question=row["question"],
            mode=row.get("mode"),
            success=row["success"],
            sql=row.get("sql"),
            attempt_count=row.get("attempt_count", 0),
            outcomes=list(row.get("outcomes") or []),
            row_count=row.get("row_count"),
            result_hash=row.get("result_hash"),
            duration_ms=row.get("duration_ms"),
        )
        expected = compute_entry_hash(prev, canonical_payload(record))
        if row["prev_hash"] != prev or row["entry_hash"] != expected:
            return row["id"]
        prev = row["entry_hash"]
    return None
