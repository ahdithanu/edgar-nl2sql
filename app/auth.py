"""API-key authentication: turn an X-API-Key header into a Principal.

Replaces the single shared secret (settings.query_api_key) with a real key
store in app_meta.api_keys:

- Keys are high-entropy random tokens shown to the operator EXACTLY ONCE at
  creation (see scripts/manage_keys.py). We persist only a SHA-256 hash and a
  short display prefix — a database leak never yields a usable key.
- Because a key is high-entropy (not a human-chosen password), a fast hash
  (SHA-256) is the correct choice: there is nothing to brute-force, so the
  slow password hashes (bcrypt/argon2) would only add latency to every request.
- Each key carries scopes, an optional per-key rate-limit override, and
  active/revoked/expiry state, all enforced at resolution time.
- A nullable tenant_id is stored now so multi-tenant isolation can be layered
  on later without a schema change; it is not yet enforced.

resolve_principal FAILS CLOSED: any lookup error, or an inactive/expired/
revoked key, yields no principal (the caller returns 401). The only
unauthenticated path is the explicitly-enabled anonymous principal, which the
HTTP layer applies when no key is presented.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets
from dataclasses import dataclass, field

from app.control_db import control_connection
from app.logging_config import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "edgk_"  # human-recognizable marker: "edgar key"
_DISPLAY_PREFIX_LEN = 12  # chars of the key stored in plaintext for identification
ANONYMOUS_LABEL = "anonymous"
LEGACY_LABEL = "legacy-shared-key"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller behind a request.

    id is the api_keys UUID for a real key, or None for the anonymous/legacy
    principals (which have no row). label is always safe to log and is
    denormalized into the audit record so it survives key deletion.
    """

    label: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    id: str | None = None
    tenant_id: str | None = None
    rate_limit_per_minute: int | None = None  # None = use the deployment default
    is_anonymous: bool = False

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


def generate_api_key() -> tuple[str, str, str]:
    """Mint a new key. Returns (full_key, display_prefix, key_hash).

    full_key is returned to the operator ONCE and never stored. Only the prefix
    and hash are persisted.
    """
    full_key = _KEY_PREFIX + secrets.token_urlsafe(32)
    return full_key, full_key[:_DISPLAY_PREFIX_LEN], hash_key(full_key)


def hash_key(full_key: str) -> str:
    """SHA-256 hex of a key. Deterministic — used for both storage and lookup."""
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


def anonymous_principal(rate_limit_per_minute: int) -> Principal:
    """The built-in principal for unauthenticated access (when enabled)."""
    return Principal(
        label=ANONYMOUS_LABEL,
        scopes=frozenset({"query"}),
        rate_limit_per_minute=rate_limit_per_minute,
        is_anonymous=True,
    )


def resolve_principal(full_key: str) -> Principal | None:
    """Look up a presented key and return its Principal, or None.

    None is returned for a missing/inactive/revoked/expired key AND for any
    database error (fail closed). The caller must treat None as unauthenticated.
    A successful lookup best-effort stamps last_used_at (never fatal).
    """
    key_hash = hash_key(full_key)
    try:
        with control_connection() as conn:
            row = conn.execute(
                """
                SELECT id, label, tenant_id, scopes, rate_limit_per_minute,
                       active, revoked_at, expires_at
                FROM app_meta.api_keys
                WHERE key_hash = %s
                """,
                (key_hash,),
            ).fetchone()
            conn.rollback()  # read-only; return the connection clean
    except Exception as exc:  # noqa: BLE001 — auth must fail closed, never 500
        logger.error("api_key_lookup_failed", error=str(exc))
        return None

    if row is None:
        return None

    now = dt.datetime.now(dt.timezone.utc)
    if not row["active"] or row["revoked_at"] is not None:
        return None
    if row["expires_at"] is not None and row["expires_at"] <= now:
        return None

    _touch_last_used(row["id"])
    return Principal(
        id=str(row["id"]),
        label=row["label"],
        tenant_id=str(row["tenant_id"]) if row["tenant_id"] is not None else None,
        scopes=frozenset(row["scopes"] or ()),
        rate_limit_per_minute=row["rate_limit_per_minute"],
    )


def _touch_last_used(key_id) -> None:
    """Record that a key was used. Best-effort: a failure never blocks a request."""
    try:
        with control_connection() as conn:
            conn.execute(
                "UPDATE app_meta.api_keys SET last_used_at = now() WHERE id = %s",
                (key_id,),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("api_key_touch_failed", error=str(exc))
