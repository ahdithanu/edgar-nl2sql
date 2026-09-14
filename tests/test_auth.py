"""Unit tests for app/auth.py — key generation, hashing, principal resolution.

The control-plane database is mocked (auth.control_connection) so these run with
no network. They assert the security-relevant behavior: keys hash deterministically
and are never stored in plaintext, and resolution FAILS CLOSED on any reason a key
should not be honored (inactive, revoked, expired, unknown, or a DB error).
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager

import pytest

import app.auth as auth
from app.auth import (
    Principal,
    anonymous_principal,
    generate_api_key,
    hash_key,
    resolve_principal,
)


def _row(**overrides) -> dict:
    base = {
        "id": "11111111-1111-1111-1111-111111111111",
        "label": "acme",
        "tenant_id": None,
        "scopes": ["query"],
        "rate_limit_per_minute": None,
        "active": True,
        "revoked_at": None,
        "expires_at": None,
    }
    base.update(overrides)
    return base


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def execute(self, query, params=None):
        # Only the SELECT against api_keys returns a row; UPDATE (touch) returns none.
        if "api_keys" in query and query.strip().upper().startswith("SELECT"):
            return _FakeResult(self._row)
        return _FakeResult(None)

    def rollback(self):
        pass

    def commit(self):
        pass


def _patch_conn(monkeypatch, row):
    @contextmanager
    def fake_control_connection():
        yield _FakeConn(row)

    monkeypatch.setattr(auth, "control_connection", fake_control_connection)


# --- key generation / hashing ----------------------------------------------


def test_generate_api_key_shape_and_hash_match():
    full, prefix, key_hash = generate_api_key()
    assert full.startswith("edgk_")
    assert full.startswith(prefix)  # prefix is a leading slice of the real key
    assert key_hash == hash_key(full)
    # The stored prefix is short — not enough to reconstruct the key.
    assert len(prefix) < len(full)


def test_hash_key_is_deterministic_and_not_the_key():
    full, _, _ = generate_api_key()
    assert hash_key(full) == hash_key(full)
    assert full not in hash_key(full)


def test_generated_keys_are_unique():
    assert generate_api_key()[0] != generate_api_key()[0]


# --- resolution: the happy path --------------------------------------------


def test_resolve_valid_key_returns_principal(monkeypatch):
    _patch_conn(monkeypatch, _row(scopes=["query", "admin"], rate_limit_per_minute=99))
    p = resolve_principal("edgk_whatever")
    assert isinstance(p, Principal)
    assert p.label == "acme"
    assert p.has_scope("query") and p.has_scope("admin")
    assert p.rate_limit_per_minute == 99
    assert p.is_anonymous is False


# --- resolution FAILS CLOSED ------------------------------------------------


def test_resolve_unknown_key_returns_none(monkeypatch):
    _patch_conn(monkeypatch, None)
    assert resolve_principal("edgk_nope") is None


def test_resolve_revoked_key_returns_none(monkeypatch):
    _patch_conn(monkeypatch, _row(revoked_at=dt.datetime.now(dt.timezone.utc)))
    assert resolve_principal("edgk_x") is None


def test_resolve_inactive_key_returns_none(monkeypatch):
    _patch_conn(monkeypatch, _row(active=False))
    assert resolve_principal("edgk_x") is None


def test_resolve_expired_key_returns_none(monkeypatch):
    past = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    _patch_conn(monkeypatch, _row(expires_at=past))
    assert resolve_principal("edgk_x") is None


def test_resolve_future_expiry_is_valid(monkeypatch):
    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
    _patch_conn(monkeypatch, _row(expires_at=future))
    assert resolve_principal("edgk_x") is not None


def test_resolve_db_error_fails_closed(monkeypatch):
    @contextmanager
    def boom():
        raise RuntimeError("control DB down")
        yield  # pragma: no cover

    monkeypatch.setattr(auth, "control_connection", boom)
    assert resolve_principal("edgk_x") is None


# --- anonymous principal ----------------------------------------------------


def test_anonymous_principal_has_query_scope_and_flag():
    p = anonymous_principal(5)
    assert p.is_anonymous is True
    assert p.has_scope("query")
    assert p.rate_limit_per_minute == 5
    assert p.id is None
