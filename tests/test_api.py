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
from app.main import app


@pytest.fixture
def client(monkeypatch, make_query_response):
    """TestClient with the pipeline and DB health check mocked out."""

    def fake_pipeline(question: str, request_id: str):
        return make_query_response(question, request_id)

    monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "check_health", lambda: True)
    # Lifespan runs configure_logging (harmless) and close_pool on exit
    # (no-op: the lazy pool singleton was never created in unit tests).
    with TestClient(app) as test_client:
        yield test_client


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

    # raise_server_exceptions=False lets the app's own Exception handler
    # produce the response instead of the test client re-raising.
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/query", json={"question": "trigger the handler"})

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

    settings = get_settings().model_copy(update={"query_api_key": "topsecret"})
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "check_health", lambda: True)

    with TestClient(app) as client:
        page = client.get("/")
        gated = client.post("/query", json={"question": "needs a key now"})

    assert page.status_code == 200
    assert gated.status_code == 401


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
