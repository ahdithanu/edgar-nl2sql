-- ============================================================================
-- P0 security build: API-key auth + tamper-evident audit log
-- ============================================================================
-- Creates the CONTROL PLANE, isolated from the public data tables:
--   * app_meta.api_keys   — hashed API keys (credential store)
--   * app_meta.audit_log  — append-only, hash-chained record of every /query
--
-- Isolation properties this migration establishes:
--   1. app_meta is a separate schema. The SQL guard (app/sql_guard.py) blocks
--      it, so LLM-generated SQL can never name these tables.
--   2. The application role gets INSERT on audit_log but NOT UPDATE/DELETE —
--      the app literally cannot alter or erase a past audit entry (append-only
--      enforced by Postgres, not just by convention).
--   3. anon/authenticated (the public PostgREST roles) get no access at all.
--
-- Idempotent. Apply as a privileged role (postgres) — e.g. Supabase SQL Editor.
--
-- NOTE (P1 hardening, documented not yet applied): for full plane separation,
-- create a dedicated login role `edgar_ctl` with access to app_meta ONLY, point
-- CONTROL_DATABASE_URL at it, and REVOKE app_meta from edgar_app. Then the
-- data-plane role that runs generated SQL has zero grant on the control plane.
-- Template at the bottom of this file.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS app_meta;

-- --- credential store -------------------------------------------------------
CREATE TABLE IF NOT EXISTS app_meta.api_keys (
    id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    label                  text NOT NULL,
    tenant_id              uuid,                         -- reserved for multi-tenancy (not yet enforced)
    key_prefix             text NOT NULL,                -- first chars, for identification in UIs/logs
    key_hash               text NOT NULL UNIQUE,         -- SHA-256 of the full key; the key itself is never stored
    scopes                 text[] NOT NULL DEFAULT ARRAY['query'],
    rate_limit_per_minute  int,                          -- per-key override; NULL = deployment default
    active                 boolean NOT NULL DEFAULT true,
    created_at             timestamptz NOT NULL DEFAULT now(),
    last_used_at           timestamptz,
    revoked_at             timestamptz,
    expires_at             timestamptz
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON app_meta.api_keys (key_hash);

-- --- audit log (append-only, hash-chained) ----------------------------------
CREATE TABLE IF NOT EXISTS app_meta.audit_log (
    id             bigserial PRIMARY KEY,                -- monotonic chain order
    request_id     text NOT NULL,
    occurred_at    timestamptz NOT NULL,
    principal_id   uuid,                                 -- NULL for anonymous/legacy
    principal_label text NOT NULL,                       -- denormalized; survives key deletion
    client_ip      text,
    question       text NOT NULL,
    mode           text,                                 -- 'sql' | 'narrative'
    success        boolean NOT NULL,
    sql            text,
    attempt_count  int NOT NULL DEFAULT 0,
    outcomes       text[],
    row_count      int,
    result_hash    text,                                 -- SHA-256 of returned rows (content proof, not the rows)
    duration_ms    int,
    prev_hash      text NOT NULL,                        -- previous row's entry_hash
    entry_hash     text NOT NULL                         -- SHA-256(prev_hash || canonical(this row))
);
CREATE INDEX IF NOT EXISTS idx_audit_request   ON app_meta.audit_log (request_id);
CREATE INDEX IF NOT EXISTS idx_audit_occurred  ON app_meta.audit_log (occurred_at);
CREATE INDEX IF NOT EXISTS idx_audit_principal ON app_meta.audit_log (principal_id);

-- --- grants: application role (edgar_app) ------------------------------------
-- The public API roles get NOTHING on the control plane.
REVOKE ALL ON SCHEMA app_meta FROM anon, authenticated;
GRANT  USAGE ON SCHEMA app_meta TO edgar_app;

-- api_keys: the app reads keys and stamps last_used_at; the CLI creates/revokes.
GRANT SELECT, INSERT, UPDATE ON app_meta.api_keys TO edgar_app;

-- audit_log: INSERT only. No UPDATE/DELETE grant => the app cannot tamper.
GRANT SELECT, INSERT ON app_meta.audit_log TO edgar_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA app_meta TO edgar_app;

-- Make sure the public API roles can't reach the new tables via default grants.
REVOKE ALL ON ALL TABLES IN SCHEMA app_meta FROM anon, authenticated;

-- ============================================================================
-- Verify:
--   SELECT table_name FROM information_schema.tables WHERE table_schema='app_meta';
--   -- edgar_app must have INSERT but NOT UPDATE/DELETE on audit_log:
--   SELECT privilege_type FROM information_schema.role_table_grants
--     WHERE table_schema='app_meta' AND table_name='audit_log' AND grantee='edgar_app';
-- ============================================================================

-- ----------------------------------------------------------------------------
-- P1 HARDENING TEMPLATE — dedicated control-plane role (run later, with a
-- generated password, then set CONTROL_DATABASE_URL and redeploy):
--   CREATE ROLE edgar_ctl LOGIN PASSWORD '<generated>';
--   GRANT USAGE ON SCHEMA app_meta TO edgar_ctl;
--   GRANT SELECT, INSERT, UPDATE ON app_meta.api_keys TO edgar_ctl;
--   GRANT SELECT, INSERT ON app_meta.audit_log TO edgar_ctl;
--   GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA app_meta TO edgar_ctl;
--   REVOKE ALL ON SCHEMA app_meta FROM edgar_app;
--   REVOKE ALL ON ALL TABLES IN SCHEMA app_meta FROM edgar_app;
-- (After this, generated SQL's role has zero reach into the control plane.)
-- ----------------------------------------------------------------------------
