"""Unit tests for app/audit.py — the tamper-evident hash chain.

The DB write path is integration-tested elsewhere; here we test the pure logic
that gives the audit log its value: a canonical serialization, a hash chain that
links each row to the previous one, and a verifier that pinpoints the first row
where the chain was broken. No database required.
"""

from __future__ import annotations

import datetime as dt

from app.audit import (
    GENESIS_HASH,
    AuditRecord,
    canonical_payload,
    compute_entry_hash,
    result_hash,
    verify_chain,
)


def _record(i: int, **overrides) -> AuditRecord:
    base = dict(
        request_id=f"req-{i}",
        principal_label="acme",
        principal_id="00000000-0000-0000-0000-000000000001",
        client_ip="203.0.113.7",
        question=f"question {i}",
        success=True,
        occurred_at=dt.datetime(2026, 1, 1, 12, i, 0, tzinfo=dt.timezone.utc),
        mode="sql",
        sql="SELECT 1",
        attempt_count=1,
        outcomes=["success"],
        row_count=1,
        result_hash="abc",
        duration_ms=42,
    )
    base.update(overrides)
    return AuditRecord(**base)


def _row_from(record: AuditRecord, row_id: int, prev_hash: str) -> dict:
    """Serialize a record to a persisted-row dict, as write_audit would."""
    entry_hash = compute_entry_hash(prev_hash, canonical_payload(record))
    return {
        "id": row_id,
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
    }


def _chain(n: int) -> list[dict]:
    rows: list[dict] = []
    prev = GENESIS_HASH
    for i in range(n):
        row = _row_from(_record(i), row_id=i + 1, prev_hash=prev)
        rows.append(row)
        prev = row["entry_hash"]
    return rows


# --- canonical serialization / hashing --------------------------------------


def test_canonical_payload_is_deterministic():
    r = _record(1)
    assert canonical_payload(r) == canonical_payload(r)


def test_entry_hash_depends_on_prev_hash():
    canon = canonical_payload(_record(1))
    assert compute_entry_hash("A", canon) != compute_entry_hash("B", canon)


def test_result_hash_changes_with_rows():
    assert result_hash([{"value": 1}]) != result_hash([{"value": 2}])
    assert result_hash([{"value": 1}]) == result_hash([{"value": 1}])


# --- chain verification -----------------------------------------------------


def test_intact_chain_verifies():
    assert verify_chain(_chain(5)) is None


def test_empty_chain_verifies():
    assert verify_chain([]) is None


def test_tampered_field_breaks_chain_at_that_row():
    rows = _chain(5)
    rows[2]["question"] = "TAMPERED"  # edit content without recomputing hashes
    assert verify_chain(rows) == rows[2]["id"]


def test_deleted_row_breaks_chain():
    rows = _chain(5)
    del rows[2]  # removing a link orphans the next row's prev_hash
    assert verify_chain(rows) == rows[2]["id"]  # now the old row 4


def test_reordered_rows_break_chain():
    rows = _chain(5)
    rows[1], rows[2] = rows[2], rows[1]
    assert verify_chain(rows) is not None
