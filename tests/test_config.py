"""Unit tests for app/config.py — settings loading and normalization."""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.mark.parametrize(
    "raw",
    [
        " sk-ant-api03-example",  # leading space (the one that actually bit us)
        "sk-ant-api03-example ",  # trailing space
        "\tsk-ant-api03-example\n",  # tab / newline from a piped value
    ],
)
def test_credentials_are_stripped_of_surrounding_whitespace(raw):
    """A secret with stray whitespace must not reach an HTTP header.

    Regression test for a genuinely nasty CI-only failure: a GitHub secret was
    stored with a leading space. Locally everything worked, because
    pydantic-settings strips values it reads from a .env file — but in CI the
    value arrived as a raw environment variable, went into the `x-api-key`
    header, and httpx rejected it with
    `LocalProtocolError: Illegal header value`. The Anthropic SDK surfaced that
    as `APIConnectionError: Connection error.`, which sent us hunting for a
    network problem that did not exist. Normalizing at the settings boundary
    makes the whole class of failure impossible.
    """
    settings = Settings(
        database_url="postgresql://u:p@127.0.0.1:5432/db",
        anthropic_api_key=raw,
        voyage_api_key=raw,
    )

    assert settings.anthropic_api_key == "sk-ant-api03-example"
    assert settings.voyage_api_key == "sk-ant-api03-example"
    # No leading/trailing whitespace survives — this is what httpx chokes on.
    assert settings.anthropic_api_key == settings.anthropic_api_key.strip()


def test_database_url_is_stripped():
    settings = Settings(database_url="  postgresql://u:p@127.0.0.1:5432/db\n")

    assert settings.database_url == "postgresql://u:p@127.0.0.1:5432/db"


def test_internal_whitespace_is_preserved():
    """Only the ends are trimmed — a value's interior is never rewritten."""
    settings = Settings(
        database_url="postgresql://u:p@127.0.0.1:5432/db",
        claude_model="  claude sonnet 5  ",
    )

    assert settings.claude_model == "claude sonnet 5"
