"""Unit tests for scripts/load_transcripts.py pure helpers.

The networked parts (FMP fetch, Voyage embed, DB write) are integration
concerns; here we test the logic that must be correct regardless of the source:
tolerant parsing of FMP's shape variants, chunking, and the synthetic ids that
make transcripts idempotent and retrievable. No network, no database.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "load_transcripts.py"
_spec = importlib.util.spec_from_file_location("load_transcripts", _SCRIPT)
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)


# --- synthetic ids ----------------------------------------------------------


def test_synthetic_accession_is_stable_and_uppercased():
    assert lt.synthetic_accession("aapl", 2024, 1) == "AAPL-2024-Q1"
    assert lt.synthetic_accession("AAPL", 2024, 1) == "AAPL-2024-Q1"


def test_section_key():
    assert lt.section_key(3) == "earnings_q3"


# --- normalize_transcript: tolerant of FMP shape variants -------------------


def test_normalize_transcript_from_list():
    payload = [{"symbol": "AAPL", "year": 2024, "quarter": 1, "date": "2024-02-01", "content": "Hello."}]
    out = lt.normalize_transcript(payload)
    assert out == {"year": 2024, "quarter": 1, "date": "2024-02-01", "content": "Hello."}


def test_normalize_transcript_from_bare_object_with_period_string():
    payload = {"year": "2023", "period": "Q4", "content": "  Remarks.  "}
    out = lt.normalize_transcript(payload)
    assert out["year"] == 2023
    assert out["quarter"] == 4
    assert out["content"] == "Remarks."


def test_normalize_transcript_empty_content_is_none():
    assert lt.normalize_transcript([{"year": 2024, "quarter": 1, "content": "   "}]) is None
    assert lt.normalize_transcript([]) is None
    assert lt.normalize_transcript("nonsense") is None


# --- normalize_available_dates: newest-first, shape-tolerant ----------------


def test_normalize_available_dates_array_shape_sorted_desc():
    payload = [[1, 2024, "2024-02-01"], [3, 2023, "2023-08-01"], [2, 2024, "2024-05-01"]]
    assert lt.normalize_available_dates(payload) == [(2024, 2), (2024, 1), (2023, 3)]


def test_normalize_available_dates_dict_shape():
    payload = [{"quarter": 2, "year": 2025}, {"period": 1, "year": 2025}]
    assert lt.normalize_available_dates(payload) == [(2025, 2), (2025, 1)]


def test_normalize_available_dates_drops_malformed():
    payload = [[1, 2024], ["x", "y", "z"], {"year": 2024}]  # 2nd/3rd malformed
    assert lt.normalize_available_dates(payload) == [(2024, 1)]


# --- chunking ---------------------------------------------------------------


def test_chunk_transcript_overlap_and_cap():
    text = "word " * 20000  # long transcript
    chunks = lt.chunk_transcript(text)
    assert 1 < len(chunks) <= lt.MAX_CHUNKS_PER_CALL
    # consecutive chunks overlap by CHUNK_OVERLAP characters
    assert chunks[0][-lt.CHUNK_OVERLAP:] == chunks[1][: lt.CHUNK_OVERLAP]


def test_chunk_transcript_short_text_single_chunk():
    assert lt.chunk_transcript("just a short call") == ["just a short call"]


def test_chunk_transcript_empty():
    assert lt.chunk_transcript("   ") == []
