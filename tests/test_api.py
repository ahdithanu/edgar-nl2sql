"""Unit tests for app/main.py — the HTTP transport layer.

The API layer's job is transport: validation, request-id plumbing, headers,
serialization, and never leaking internals. The pipeline is mocked with a
canned QueryResponse (see conftest.make_query_response) so these tests fail
only when the HTTP contract breaks, not when pipeline logic changes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.auth import Principal
from app.main import app, require_principal


def _test_principal(**overrides) -> Principal:
    defaults = dict(label="test-key", scopes=frozenset({"query"}), id="test-id")
    defaults.update(overrides)
    return Principal(**defaults)


@pytest.fixture
def client(monkeypatch, make_query_response):
    """TestClient with the pipeline, DB health, auth, and audit mocked out.

    Auth is bypassed via FastAPI's dependency_overrides (a resolved principal is
    injected) so these transport tests don't need a control-plane database; the
    real auth logic is exercised in test_auth.py and the API tests below that
    build their own client without the override.
    """

    def fake_pipeline(question: str, request_id: str):
        return make_query_response(question, request_id)

    monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "check_health", lambda: True)
    monkeypatch.setattr(main, "_write_query_audit", lambda *a, **k: None)
    app.dependency_overrides[require_principal] = lambda: _test_principal()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


def test_query_success(client):
    resp = client.post("/query", json={"question": "What was Apple's 2023 revenue?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["question"] == "What was Apple's 2023 revenue?"
    assert body["sql"].startswith("SELECT")
    assert body["rows"] == [{"value": 391035000000.0}]
    assert body["attempts"][0]["outcome"] == "success"


def test_query_returns_request_id_in_header_and_body(client):
    resp = client.post("/query", json={"question": "What was Apple's 2023 revenue?"})

    header_id = resp.headers.get("X-Request-ID")
    assert header_id  # middleware always sets it
    # The id the middleware minted is the same one the pipeline received and
    # echoed into the body — one correlation id end to end.
    assert resp.json()["request_id"] == header_id


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing field
        {"question": "hi"},  # under min_length=3
        {"question": "x" * 1001},  # over max_length=1000
    ],
)
def test_query_validation_errors_return_422_envelope(client, payload):
    resp = client.post("/query", json=payload)

    assert resp.status_code == 422
    body = resp.json()
    # Consistent error envelope: field details + request_id, never a trace.
    assert "detail" in body
    assert "request_id" in body
    assert resp.headers.get("X-Request-ID")


def test_unhandled_exception_returns_opaque_500(monkeypatch, make_query_response):
    def exploding_pipeline(question: str, request_id: str):
        raise RuntimeError("secret internal detail: /etc/passwd")

    monkeypatch.setattr(main, "run_pipeline", exploding_pipeline)
    monkeypatch.setattr(main, "check_health", lambda: True)
    monkeypatch.setattr(main, "_write_query_audit", lambda *a, **k: None)
    app.dependency_overrides[require_principal] = lambda: _test_principal()

    # raise_server_exceptions=False lets the app's own Exception handler
    # produce the response instead of the test client re-raising.
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/query", json={"question": "trigger the handler"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "Internal server error."
    assert "request_id" in body
    # The whole point: internals never reach the client.
    assert "secret internal detail" not in resp.text
    assert "Traceback" not in resp.text
    assert resp.headers.get("X-Request-ID")


# ---------------------------------------------------------------------------
# GET / (demo page)
# ---------------------------------------------------------------------------


def test_demo_page_served_at_root(client):
    resp = client.get("/")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "edgar-" in resp.text  # the page title


def test_demo_page_is_not_behind_api_key(monkeypatch, make_query_response):
    """The static page stays public even when /query requires an API key."""
    from app.config import get_settings

    # Anonymous access off (the secure default) => /query needs a key, / does not.
    settings = get_settings().model_copy(
        update={"query_api_key": "topsecret", "anonymous_principal_enabled": False}
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "check_health", lambda: True)

    with TestClient(app) as client:
        page = client.get("/")
        gated = client.post("/query", json={"question": "needs a key now"})

    assert page.status_code == 200
    assert gated.status_code == 401


# ---------------------------------------------------------------------------
# POST /query — authentication, authorization, rate limiting
# ---------------------------------------------------------------------------


def test_query_401_without_key_when_anonymous_disabled(monkeypatch):
    """Secure default: no key + anonymous off => 401 with the standard envelope."""
    from app.config import get_settings

    settings = get_settings().model_copy(
        update={"query_api_key": "", "anonymous_principal_enabled": False}
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "check_health", lambda: True)

    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "no key at all"})

    assert resp.status_code == 401
    body = resp.json()
    assert body["detail"] == "Invalid or missing API key."
    assert "request_id" in body
    assert resp.headers.get("X-Request-ID")


def test_query_legacy_shared_key_is_accepted(monkeypatch, make_query_response):
    """A request whose X-API-Key matches the legacy secret is served."""
    from app.config import get_settings

    settings = get_settings().model_copy(update={"query_api_key": "topsecret"})
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "check_health", lambda: True)
    monkeypatch.setattr(main, "_write_query_audit", lambda *a, **k: None)
    monkeypatch.setattr(
        main, "run_pipeline", lambda q, rid: make_query_response(q, rid)
    )

    with TestClient(app) as client:
        ok = client.post(
            "/query", json={"question": "with the key"}, headers={"X-API-Key": "topsecret"}
        )
        bad = client.post(
            "/query", json={"question": "wrong key"}, headers={"X-API-Key": "nope"}
        )

    assert ok.status_code == 200
    assert bad.status_code == 401


def test_query_403_when_scope_missing(client):
    """A principal without the 'query' scope is forbidden."""
    app.dependency_overrides[require_principal] = lambda: _test_principal(
        scopes=frozenset()
    )
    resp = client.post("/query", json={"question": "no scope"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "This API key lacks the 'query' scope."


def test_query_429_when_rate_limited(client):
    """Per-principal cap: the second call in the window is rejected with 429."""
    app.dependency_overrides[require_principal] = lambda: _test_principal(
        id="rate-test", rate_limit_per_minute=1
    )
    first = client.post("/query", json={"question": "first call"})
    second = client.post("/query", json={"question": "second call"})
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers.get("Retry-After") == "60"
    assert second.headers.get("X-Request-ID")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_ok_when_db_up(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["version"] == "0.1.0"


def test_health_degraded_but_still_200_when_db_down(monkeypatch, make_query_response):
    monkeypatch.setattr(main, "check_health", lambda: False)

    with TestClient(app) as client:
        resp = client.get("/health")

    # Always HTTP 200: a DB outage is reported in the body, not as an error
    # status — orchestrators must not restart-loop a healthy app.
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["database"] is False
