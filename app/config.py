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
    # Control-plane DSN for the credential store + audit log (schema app_meta).
    # Optional: falls back to database_url when unset (see control_dsn). In a
    # hardened deployment this is a SEPARATE least-privileged role (edgar_ctl)
    # that can touch app_meta but NOT the public data tables, so the data-plane
    # role that runs LLM-generated SQL can never reach credentials or audit rows.
    control_database_url: str = ""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    # --- model selection ---
    claude_model: str = "claude-sonnet-5"
    embed_model: str = "voyage-3.5-lite"  # 1024-dim; must match VECTOR(1024) in schema.sql

    # --- API protection ---
    # POST /query is expensive per call (1 Voyage embed + up to 3 Claude
    # generations + synthesis) and holds a pooled DB connection, so it must
    # not be an unauthenticated, unmetered cost sink when exposed publicly.
    query_api_key: str = ""  # LEGACY single shared secret; superseded by the
    # app_meta.api_keys table (see app/auth.py). Still honored for backward
    # compatibility: a request whose X-API-Key matches is accepted as the
    # "legacy" principal. Leave unset once real keys are provisioned.
    rate_limit_per_minute: int = 30  # default per-principal cap on POST /query; 0 disables

    # --- authentication / access ---
    # When True, a /query request with NO (or an unrecognized) API key is served
    # as the built-in "anonymous" principal instead of being rejected with 401.
    # This is what keeps the PUBLIC demo working. Secure-by-default is OFF: a
    # deployment must explicitly opt in to unauthenticated access.
    anonymous_principal_enabled: bool = False
    anonymous_rate_limit_per_minute: int = 5  # tight cap for the anonymous principal
    # Append a tamper-evident row to app_meta.audit_log for every /query. Never
    # fails the request — an audit-write error is logged, not surfaced.
    audit_enabled: bool = True

    @field_validator(
        "database_url",
        "control_database_url",
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

    @property
    def control_dsn(self) -> str:
        """DSN for the control plane (credential store + audit log).

        Falls back to the data-plane DSN when control_database_url is unset, so
        a single-role deployment works out of the box. Set control_database_url
        to a dedicated least-privileged role to fully isolate the planes.
        """
        return self.control_database_url or self.database_url


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    WHY lru_cache: settings are immutable for the life of the process, and caching lets
    tests swap configuration by calling `get_settings.cache_clear()` after patching the
    environment — no global mutable state to reset.
    """
    return Settings()
