"""Keep fresh initialization and migrations aligned without a live database."""

from pathlib import Path

import pytest
from sqlglot import parse, parse_one


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("filename", ["schema.sql", "003_filing_chunks_loaded_at.sql"])
def test_filing_chunks_recency_index(filename):
    """Both entry points must create the same idempotent descending index."""
    expected = parse_one(
        "CREATE INDEX IF NOT EXISTS idx_filing_chunks_loaded_at "
        "ON filing_chunks (loaded_at DESC)",
        read="postgres",
    )
    statements = parse((SCRIPTS / filename).read_text(), read="postgres")
    assert expected in statements, f"{filename} is missing the recency index definition"
