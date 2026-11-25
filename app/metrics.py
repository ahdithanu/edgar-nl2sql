"""Per-request latency + token/cost accounting.

The pipeline makes several paid calls per question (one Voyage embed + up to
three Claude generations + a synthesis). Nothing recorded how long they took or
what they cost, so "p95 latency" and "$/query" were unanswerable. This module
captures both, keyed to the request via a contextvar (same pattern as the
request_id log binding), so generation.py and retrieval.py record into it
without threading an object through every signature.

Token counts are MEASURED (read from each provider's `usage`/`total_tokens`).
The dollar figure is those measured tokens times a configurable price table —
so cost is "measured tokens x assumed rate". Claude rates are the published
Anthropic prices; the Voyage rate is a documented estimate (embeddings are a
tiny fraction of per-query cost, which is dominated by Claude output tokens).
Update PRICES if rates change.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field

# USD per 1,000,000 tokens: (input, output). Keep in sync with provider pricing.
LLM_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
_DEFAULT_LLM_PRICE = (2.00, 10.00)  # fall back to Sonnet-class if model unknown

# USD per 1,000,000 tokens for embeddings. voyage-3.5-lite is inexpensive;
# verify against current Voyage pricing. Negligible next to Claude tokens.
EMBED_PRICES: dict[str, float] = {"voyage-3.5-lite": 0.02}
_DEFAULT_EMBED_PRICE = 0.02


@dataclass
class _LLMCall:
    model: str
    input_tokens: int
    output_tokens: int
    ms: int


@dataclass
class _EmbedCall:
    model: str
    tokens: int
    ms: int


@dataclass
class RequestMetrics:
    """Accumulates the LLM + embed calls made while serving one request."""

    llm_calls: list[_LLMCall] = field(default_factory=list)
    embed_calls: list[_EmbedCall] = field(default_factory=list)
    total_ms: int | None = None  # wall time, stamped by the pipeline wrapper

    def record_llm(self, model: str, input_tokens: int, output_tokens: int, ms: int) -> None:
        self.llm_calls.append(_LLMCall(model, input_tokens, output_tokens, ms))

    def record_embed(self, model: str, tokens: int, ms: int) -> None:
        self.embed_calls.append(_EmbedCall(model, tokens, ms))

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def embed_tokens(self) -> int:
        return sum(c.tokens for c in self.embed_calls)

    @property
    def llm_ms(self) -> int:
        return sum(c.ms for c in self.llm_calls)

    def est_cost_usd(self) -> float:
        cost = 0.0
        for c in self.llm_calls:
            price_in, price_out = LLM_PRICES.get(c.model, _DEFAULT_LLM_PRICE)
            cost += (c.input_tokens * price_in + c.output_tokens * price_out) / 1_000_000
        for e in self.embed_calls:
            cost += e.tokens * EMBED_PRICES.get(e.model, _DEFAULT_EMBED_PRICE) / 1_000_000
        return cost

    def summary(self) -> dict:
        """Flat, log/JSON-friendly snapshot (keys become structured-log fields)."""
        return {
            "llm_calls": len(self.llm_calls),
            "embed_calls": len(self.embed_calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "embed_tokens": self.embed_tokens,
            "llm_ms": self.llm_ms,
            "est_cost_usd": round(self.est_cost_usd(), 6),
        }


# Request-scoped current accumulator. None outside a request (e.g. the loaders),
# so record_* become no-ops there rather than requiring every caller to opt in.
_current: contextvars.ContextVar[RequestMetrics | None] = contextvars.ContextVar(
    "request_metrics", default=None
)


def start_metrics() -> RequestMetrics:
    """Begin a fresh accumulator for the current request/context."""
    m = RequestMetrics()
    _current.set(m)
    return m


def current_metrics() -> RequestMetrics | None:
    return _current.get()


def record_llm(model: str, input_tokens: int, output_tokens: int, ms: int) -> None:
    m = _current.get()
    if m is not None:
        m.record_llm(model, input_tokens, output_tokens, ms)


def record_embed(model: str, tokens: int, ms: int) -> None:
    m = _current.get()
    if m is not None:
        m.record_embed(model, tokens, ms)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0,100]); 0.0 for an empty list.

    Nearest-rank (not interpolated) so a p95 is always an actual observed
    sample — the honest thing to quote for a small N.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(1, min(len(ordered), -(-len(ordered) * int(pct) // 100)))  # ceil
    return ordered[k - 1]
