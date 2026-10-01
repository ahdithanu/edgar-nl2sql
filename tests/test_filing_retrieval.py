"""Unit tests for app/filing_retrieval._describe_chunk (citation labeling).

One retrieval path serves both 10-K sections and earnings-call transcripts; the
citation must be derived correctly from `form`. The 10-K titles must stay
byte-identical to before the transcript feature, because the eval's narrative
cases assert on them. No network, no database.
"""

from __future__ import annotations

from app.filing_retrieval import _describe_chunk


def test_10k_risk_factors_label_unchanged():
    doc_type, title = _describe_chunk("10-K", "AAPL", "risk_factors", 2024)
    assert doc_type == "filing:risk_factors"
    assert title == "AAPL 10-K Risk Factors FY2024"


def test_10k_mda_label_unchanged():
    _, title = _describe_chunk("10-K", "MSFT", "mda", 2023)
    assert title == "MSFT 10-K MD&A FY2023"


def test_10k_without_fiscal_year_omits_fy():
    _, title = _describe_chunk("10-K", "NVDA", "risk_factors", None)
    assert title == "NVDA 10-K Risk Factors"


def test_earnings_call_label():
    doc_type, title = _describe_chunk("earnings_call", "AAPL", "earnings_q1", 2024)
    assert doc_type == "transcript:earnings_q1"
    assert title == "AAPL Earnings Call Q1 FY2024"


def test_unknown_section_falls_back_to_raw():
    _, title = _describe_chunk("10-K", "KO", "item8", 2022)
    assert title == "KO 10-K item8 FY2022"
