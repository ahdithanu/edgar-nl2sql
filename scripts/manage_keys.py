"""Operator CLI for API keys (app_meta.api_keys).

The full key is shown ONCE at creation and never stored — copy it then. Only a
SHA-256 hash and a short prefix live in the database.

Usage:
    python scripts/manage_keys.py create --label "acme-prod" [--scopes query] \
        [--rate-limit 60] [--expires-days 365] [--tenant <uuid>]
    python scripts/manage_keys.py list
    python scripts/manage_keys.py revoke --id <uuid>
    python scripts/manage_keys.py verify-audit        # check the audit hash chain

Connects with CONTROL_DATABASE_URL (falls back to DATABASE_URL), so it uses the
same control-plane credentials as the app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.audit import verify_chain  # noqa: E402
from app.auth import generate_api_key  # noqa: E402
from app.config import get_settings  # noqa: E402


def _connect() -> psycopg.Connection:
    return psycopg.connect(get_settings().control_dsn, row_factory=dict_row)


def cmd_create(args: argparse.Namespace) -> None:
    full_key, prefix, key_hash = generate_api_key()
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    expires = None
    if args.expires_days:
        expires = f"now() + interval '{int(args.expires_days)} days'"

    with _connect() as conn:
        conn.execute(
            f"""
            INSERT INTO app_meta.api_keys
                (label, tenant_id, key_prefix, key_hash, scopes,
                 rate_limit_per_minute, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s, {expires or 'NULL'})
            """,
            (args.label, args.tenant, prefix, key_hash, scopes, args.rate_limit),
        )
        conn.commit()

    print("API key created. Copy it now — it will NOT be shown again:\n")
    print(f"    {full_key}\n")
    print(f"label={args.label}  prefix={prefix}  scopes={scopes}")


def cmd_list(_: argparse.Namespace) -> None:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, label, key_prefix, scopes, rate_limit_per_minute,
                   active, created_at, last_used_at, revoked_at, expires_at
            FROM app_meta.api_keys ORDER BY created_at
            """
        ).fetchall()
    if not rows:
        print("(no API keys)")
        return
    for r in rows:
        state = "revoked" if r["revoked_at"] else ("active" if r["active"] else "inactive")
        print(
            f"{r['id']}  {r['label']:<20} {r['key_prefix']}...  [{state}]  "
            f"scopes={r['scopes']}  last_used={r['last_used_at']}"
        )


def cmd_revoke(args: argparse.Namespace) -> None:
    with _connect() as conn:
        n = conn.execute(
            "UPDATE app_meta.api_keys SET active=false, revoked_at=now() "
            "WHERE id=%s AND revoked_at IS NULL",
            (args.id,),
        ).rowcount
        conn.commit()
    print("revoked" if n else "no matching active key")


def cmd_verify_audit(_: argparse.Namespace) -> None:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM app_meta.audit_log ORDER BY id ASC"
        ).fetchall()
    broken = verify_chain(rows)
    if broken is None:
        print(f"audit chain intact ({len(rows)} entries)")
    else:
        print(f"AUDIT CHAIN BROKEN at row id={broken} — tampering or data loss")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage edgar-nl2sql API keys.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="mint a new API key")
    c.add_argument("--label", required=True)
    c.add_argument("--scopes", default="query", help="comma-separated (default: query)")
    c.add_argument("--rate-limit", type=int, default=None, help="per-minute override")
    c.add_argument("--expires-days", type=int, default=None)
    c.add_argument("--tenant", default=None, help="tenant UUID (optional)")
    c.set_defaults(func=cmd_create)

    sub.add_parser("list", help="list keys").set_defaults(func=cmd_list)

    r = sub.add_parser("revoke", help="revoke a key by id")
    r.add_argument("--id", required=True)
    r.set_defaults(func=cmd_revoke)

    sub.add_parser("verify-audit", help="verify the audit hash chain").set_defaults(
        func=cmd_verify_audit
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
