"""Application settings, loaded once from the environment (or a local .env file).

WHY pydantic-settings: configuration errors should fail loudly at startup with a clear
validation message (e.g. a missing DATABASE_URL), not surface later as a cryptic
connection error mid-request. Every knob that ops might want to tune (model names,
retrieval depth, row caps, statement timeout) lives here rather than being buried as a
magic number in the code.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the service.

    Values are read from environment variables (case-insensitive) with a `.env` file
    fallback for local development. Only `database_url` is strictly required; the API
    keys default to empty strings so that unit tests — which mock every external
    boundary — can import the app without any credentials present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated vars in shared .env files
    )

    # --- external services ---
    database_url: str  # Supabase pooler URL; required — no safe default exists
    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    # --- model selection ---
    claude_model: str = "claude-sonnet-5"
    embed_model: str = "voyage-3.5-lite"  # 1024-dim; must match VECTOR(1024) in schema.sql

    # --- API protection ---
    # POST /query is expensive per call (1 Voyage embed + up to 3 Claude
    # generations + synthesis) and holds a pooled DB connection, so it must
    # not be an unauthenticated, unmetered cost sink when exposed publicly.
    query_api_key: str = ""  # if set, POST /query requires a matching X-API-Key header
    rate_limit_per_minute: int = 30  # per-client-IP cap on POST /query; 0 disables

    @field_validator(
        "database_url",
        "anthropic_api_key",
        "voyage_api_key",
        "query_api_key",
        "claude_model",
        "embed_model",
        mode="before",
    )
    @classmethod
    def _strip_whitespace(cls, value: object) -> object:
        """Trim surrounding whitespace from credential/identifier values.

        WHY this exists: a secret pasted or piped with a stray leading space is
        invisible in every UI, and pydantic-settings already strips values read
        from a .env file — so it works locally and breaks only in CI/production,
        where the value arrives as a raw environment variable. The failure is
        also badly misleading: httpx rejects a header whose value starts with a
        space, and the Anthropic SDK surfaces that as `APIConnectionError:
        Connection error.` — which reads like a network outage, not a typo.
        (Diagnosed exactly once, the hard way. Never again.)
        """
        return value.strip() if isinstance(value, str) else value

    # --- pipeline tuning ---
    retrieval_top_k: int = 8  # context docs injected per generation call
    max_result_rows: int = 200  # hard cap on rows returned to the client / the LLM
    statement_timeout_ms: int = 10_000  # kills runaway generated SQL server-side

    # --- observability ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    WHY lru_cache: settings are immutable for the life of the process, and caching lets
    tests swap configuration by calling `get_settings.cache_clear()` after patching the
    environment — no global mutable state to reset.
    """
    return Settings()
