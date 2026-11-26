"""Unit tests for app/metrics.py — cost math, recording, and percentile.

No network: the recorders are exercised directly. The dollar figure is the
part most likely to be quoted in a post or interview, so the arithmetic is
pinned against hand-computed values.
"""

from __future__ import annotations

import pytest

from app import metrics
from app.metrics import RequestMetrics, percentile


def test_cost_is_measured_tokens_times_rate():
    m = RequestMetrics()
    # 1,000,000 input + 1,000,000 output on sonnet-5 => $2 + $10 = $12.
    m.record_llm("claude-sonnet-5", 1_000_000, 1_000_000, ms=100)
    assert m.est_cost_usd() == pytest.approx(12.0)


def test_cost_sums_multiple_calls_and_embeds():
    m = RequestMetrics()
    m.record_llm("claude-sonnet-5", 500_000, 100_000, ms=50)   # $1.00 + $1.00
    m.record_llm("claude-sonnet-5", 500_000, 100_000, ms=50)   # $1.00 + $1.00
    m.record_embed("voyage-3.5-lite", 1_000_000, ms=10)         # $0.02
    assert m.est_cost_usd() == pytest.approx(4.02)
    assert m.input_tokens == 1_000_000
    assert m.output_tokens == 200_000
    assert m.llm_ms == 100


def test_unknown_model_falls_back_to_default_rate():
    m = RequestMetrics()
    m.record_llm("some-future-model", 1_000_000, 0, ms=1)
    assert m.est_cost_usd() == pytest.approx(2.0)  # default (sonnet-class) input rate


def test_summary_shape():
    m = RequestMetrics()
    m.record_llm("claude-sonnet-5", 100, 50, ms=5)
    s = m.summary()
    assert s["llm_calls"] == 1 and s["input_tokens"] == 100 and s["output_tokens"] == 50
    assert "est_cost_usd" in s


# --- contextvar recorders --------------------------------------------------


def test_record_helpers_are_noop_without_a_started_context():
    # Fresh interpreter state: no metrics started -> recording is a safe no-op.
    # (start_metrics sets a new one; we don't call it here.)
    metrics._current.set(None)
    metrics.record_llm("claude-sonnet-5", 100, 100, 10)  # must not raise
    assert metrics.current_metrics() is None


def test_start_metrics_then_record():
    m = metrics.start_metrics()
    metrics.record_llm("claude-sonnet-5", 10, 20, 5)
    metrics.record_embed("voyage-3.5-lite", 7, 2)
    assert metrics.current_metrics() is m
    assert m.input_tokens == 10 and m.output_tokens == 20 and m.embed_tokens == 7


# --- percentile ------------------------------------------------------------


def test_percentile_nearest_rank():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(vals, 50) == 5.0   # ceil(10*50/100)=5 -> 5th value
    assert percentile(vals, 95) == 10.0  # ceil(10*95/100)=10 -> 10th value
    assert percentile(vals, 100) == 10.0


def test_percentile_empty_is_zero():
    assert percentile([], 95) == 0.0


def test_percentile_single_value():
    assert percentile([3.5], 95) == 3.5
