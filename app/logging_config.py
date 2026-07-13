"""Structured JSON logging via structlog.

WHY structured logs: the pipeline emits one `sql_attempt` event per retry-loop iteration
with machine-readable fields (request_id, attempt_number, sql, outcome, error_message,
duration_ms). As JSON lines on stdout these are directly greppable/queryable in any log
aggregator — you can answer "what fraction of requests needed a second attempt?" with a
one-liner instead of parsing prose.

WHY contextvars: `request_id` is bound once per request (in app/main.py) via
`structlog.contextvars.bind_contextvars`, and `merge_contextvars` below stamps it onto
every event logged anywhere in that request's call stack — no need to thread the id
through every function signature.
"""

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit one JSON object per line on stdout.

    Idempotent and cheap — safe to call from the FastAPI lifespan hook on every startup
    (including test clients spinning the app up repeatedly).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            # Merge request-scoped context (request_id) into every event first, so the
            # rest of the chain — and the final JSON — always carries it.
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # Render tracebacks as strings inside the JSON payload rather than letting
            # them escape as raw multi-line text (keeps one-event-per-line invariant).
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        # Filter by level at the logger wrapper — cheapest possible early exit.
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structlog logger.

    The name is bound as a `logger` field so events from db/pipeline/api layers are
    distinguishable in aggregated output.
    """
    return structlog.get_logger(name).bind(logger=name)
