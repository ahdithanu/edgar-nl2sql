"""FastAPI entrypoint for the edgar-nl2sql service.

Deliberately thin: HTTP concerns only. All NL→SQL intelligence (retrieval,
generation, the 3-attempt self-correction loop) lives in ``app.pipeline`` so it
can be exercised and tested without a web server. This module owns:

- **Request identity**: every request gets a ``request_id`` (uuid4 hex) bound
  into structlog contextvars, so every log line emitted anywhere downstream —
  retrieval, each SQL attempt, DB errors — carries the same correlation id.
  The id is also returned in the ``X-Request-ID`` response header and in the
  response body, so a user-reported failure can be grepped straight out of
  the logs.
- **Failure containment**: a global exception handler guarantees the API never
  leaks a stack trace or internal detail to a client. (The pipeline itself is
  designed never to raise, so this handler is the belt to that suspenders.)
- **Lifecycle**: logging is configured at startup; the DB connection pool is
  closed cleanly at shutdown.

Endpoints are intentionally ``def`` (sync), not ``async def``: the stack uses
psycopg 3 in sync mode, and FastAPI runs sync endpoints in its threadpool, so
a slow query blocks one worker thread instead of the event loop.
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.audit import AuditRecord, result_hash, write_audit
from app.auth import (
    LEGACY_LABEL,
    Principal,
    anonymous_principal,
    resolve_principal,
)
from app.config import get_settings
from app.control_db import close_control_pool
from app.db import check_health, close_pool
from app.logging_config import configure_logging, get_logger
from app.models import HealthResponse, QueryRequest, QueryResponse
from app.pipeline import run_pipeline

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"


class _RateLimiter:
    """Tiny in-process sliding-window rate limiter, keyed by client IP.

    WHY: every /query call burns real money (a Voyage embed + up to 3 Claude
    completions + synthesis) and holds one of only 5 pooled DB connections, so
    an unmetered public endpoint is an unbounded-cost / pool-exhaustion DoS.
    In-process is deliberate — no Redis dependency for a single-instance demo;
    a multi-replica deployment would move this to a shared store.

    Thread safety matters: endpoints are sync ``def``, so FastAPI runs them
    concurrently in its threadpool.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, limit: int, window_s: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > window_s:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


_rate_limiter = _RateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging before serving; release DB connections on shutdown.

    Logging is configured here (not at import time) so the level comes from
    settings and so importing ``app.main`` in tests has no side effects.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "startup",
        model=settings.claude_model,
        anonymous_access=settings.anonymous_principal_enabled,
        audit_enabled=settings.audit_enabled,
    )
    yield
    # Return pooled connections gracefully so Supabase's pooler isn't left
    # holding half-open sessions after a deploy or scale-down.
    close_pool()
    close_control_pool()
    logger.info("shutdown")


app = FastAPI(
    title="edgar-nl2sql",
    description="Natural-language to SQL RAG over SEC EDGAR financial data.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Assign a request_id and bind it to structlog's contextvars.

    Binding happens HERE — before any handler code runs — so every log event
    for the lifetime of this request automatically carries ``request_id``
    without any function needing to pass it around. contextvars are
    task/thread-local, so concurrent requests never see each other's ids.
    """
    request_id = uuid.uuid4().hex
    # Clear first: threadpool threads and event-loop tasks can be reused
    # across requests, and stale bindings would mis-attribute log lines.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    # Stash on request.state so endpoints and exception handlers can reach it.
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


def _request_id_of(request: Request) -> str:
    """Best-effort request_id lookup for exception handlers.

    Starlette's exception handlers can fire outside our middleware's normal
    return path, so we fall back to a fresh id rather than crash while
    building an error response.
    """
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 with field-level details — bad input is the client's to fix.

    Overridden (rather than using FastAPI's default) so the response also
    carries the request_id, keeping the error envelope consistent.
    """
    request_id = _request_id_of(request)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Render HTTPExceptions (401/403/429/...) in the standard envelope.

    Overridden so auth/rate-limit rejections carry the request_id and the
    X-Request-ID header just like every other response, and preserve any
    headers the exception set (e.g. Retry-After on a 429).
    """
    request_id = _request_id_of(request)
    headers = dict(getattr(exc, "headers", None) or {})
    headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id},
        headers=headers,
    )


def require_principal(request: Request) -> Principal:
    """Resolve the caller to a Principal or reject with 401.

    Resolution order:
      1. X-API-Key present and matches the LEGACY shared secret -> legacy principal.
      2. X-API-Key present -> look it up in app_meta.api_keys (fails closed).
      3. No key AND anonymous access enabled -> the anonymous principal.
      4. Otherwise -> 401.

    The resolved principal is stashed on request.state and bound into the log
    context so every downstream event is attributable.
    """
    settings = get_settings()
    provided = request.headers.get(API_KEY_HEADER, "").strip()

    principal: Principal | None = None
    if provided:
        if settings.query_api_key and secrets.compare_digest(
            provided, settings.query_api_key
        ):
            principal = Principal(label=LEGACY_LABEL, scopes=frozenset({"query"}))
        else:
            principal = resolve_principal(provided)
    elif settings.anonymous_principal_enabled:
        principal = anonymous_principal(settings.anonymous_rate_limit_per_minute)

    if principal is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    request.state.principal = principal
    structlog.contextvars.bind_contextvars(
        principal=principal.label, principal_id=principal.id
    )
    return principal


def _write_query_audit(
    result: QueryResponse, principal: Principal, client_ip: str, duration_ms: int
) -> None:
    """Append a tamper-evident audit row for a served /query. Best-effort."""
    if not get_settings().audit_enabled:
        return
    write_audit(
        AuditRecord(
            request_id=result.request_id,
            principal_label=principal.label,
            principal_id=principal.id,
            client_ip=client_ip,
            question=result.question,
            mode=result.mode,
            success=result.success,
            sql=result.sql,
            attempt_count=len(result.attempts),
            outcomes=[a.outcome for a in result.attempts],
            row_count=len(result.rows),
            result_hash=result_hash(result.rows) if result.rows else None,
            duration_ms=duration_ms,
        )
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defense: log the real error, return an opaque 500.

    Stack traces and exception messages go to structured logs (where the
    request_id makes them findable) — NEVER to the client, where they could
    leak schema details, file paths, or provider error internals.
    """
    request_id = _request_id_of(request)
    logger.error(
        "unhandled_exception",
        request_id=request_id,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


_DEMO_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    """Serve the single-file demo UI.

    A plain FileResponse rather than a StaticFiles mount: there is exactly one
    asset, and mounting a directory at ``/`` would shadow the API routes.
    Deliberately outside the API-key gate — the page is free static HTML with
    zero cost surface; the /query calls it makes are still gated and
    rate-limited like any other client's.
    """
    return FileResponse(_DEMO_PAGE, media_type="text/html")


@app.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> QueryResponse:
    """Answer a natural-language financial question.

    Gatekeeping first — this endpoint is the service's entire cost surface
    (embeddings, up to 3 LLM calls, a pooled DB connection per request):

    - Authentication (``require_principal``): the caller is resolved to a
      Principal from its ``X-API-Key`` (or the anonymous principal when
      anonymous access is enabled); 401 otherwise. ``/`` and ``/health`` are
      deliberately exempt so the demo page loads and load balancers can probe.
    - Authorization: the principal must hold the ``query`` scope (403 otherwise).
    - Rate limit: per-principal (per-IP for the anonymous principal), using the
      key's own override or the deployment default; 429 when exceeded.

    Every served request — success or failure — is recorded to the tamper-
    evident audit log (best-effort; auditing never fails the request).

    The heavy lifting — retrieve context, generate SQL, validate, execute,
    self-correct up to 3 times, synthesize an answer — is ``run_pipeline``.
    It is contractually non-raising: failures come back as a structured
    ``QueryResponse`` with ``success=False`` and a readable explanation.
    """
    settings = get_settings()
    request_id: str = request.state.request_id
    client_ip = request.client.host if request.client else "unknown"

    if not principal.has_scope("query"):
        raise HTTPException(
            status_code=403, detail="This API key lacks the 'query' scope."
        )

    limit = (
        principal.rate_limit_per_minute
        if principal.rate_limit_per_minute is not None
        else settings.rate_limit_per_minute
    )
    if limit > 0:
        # Real keys are limited by identity; the anonymous principal (which has
        # no id and is shared) is limited per client IP so one abuser can't
        # exhaust the whole anonymous budget for everyone.
        rl_key = principal.id or (
            f"anon:{client_ip}" if principal.is_anonymous else f"label:{principal.label}"
        )
        if not _rate_limiter.allow(rl_key, limit):
            logger.warning("rate_limited", principal=principal.label, client_ip=client_ip)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded; retry shortly.",
                headers={"Retry-After": "60"},
            )

    started = time.monotonic()
    result = run_pipeline(payload.question, request_id)
    duration_ms = int((time.monotonic() - started) * 1000)

    _write_query_audit(result, principal, client_ip, duration_ms)
    return result


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + dependency check.

    Always HTTP 200: a load balancer probing this endpoint should learn the
    *app* is up even when the database isn't — that state is reported as
    ``status="degraded"`` in the body rather than as an error status, so
    orchestrators don't restart-loop a healthy app over a DB outage.
    """
    db_ok = check_health()
    return HealthResponse(status="ok" if db_ok else "degraded", database=db_ok)
