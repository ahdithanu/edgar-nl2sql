"""Unit tests for app/sql_guard.py — the static SQL validation gate.

These are pure tests: sqlglot parsing only, no database, no network. They
encode the guard's contract: exactly one statement, read-only SELECT at the
top level, no DML/DDL anywhere in the tree (data-modifying CTEs included),
no SELECT INTO, no system-catalog snooping, trailing semicolon stripped.
"""

from __future__ import annotations

import pytest

from app.sql_guard import SQLGuardError, validate_sql

# ---------------------------------------------------------------------------
# Accepted queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT fm.value FROM financial_metrics fm WHERE fm.metric = 'revenue'",
        # Joins + aggregates — the workhorse shape for this schema.
        (
            "SELECT c.ticker, SUM(fm.value) AS total "
            "FROM financial_metrics fm JOIN companies c ON c.id = fm.company_id "
            "WHERE fm.fiscal_year = 2023 GROUP BY c.ticker ORDER BY total DESC LIMIT 5"
        ),
        # CTEs are allowed as long as the final expression is a SELECT.
        (
            "WITH rev AS (SELECT company_id, value FROM financial_metrics "
            "WHERE metric = 'revenue') SELECT * FROM rev"
        ),
        # Subqueries and window functions are plain read-only SQL.
        (
            "SELECT ticker FROM companies WHERE id IN "
            "(SELECT company_id FROM financial_metrics WHERE metric = 'revenue')"
        ),
    ],
)
def test_accepts_read_only_selects(sql: str) -> None:
    assert validate_sql(sql) == sql


def test_strips_trailing_semicolon_and_whitespace() -> None:
    assert validate_sql("  SELECT 1;  ") == "SELECT 1"


def test_returns_the_models_own_sql_not_a_rerender() -> None:
    # The guard must not rewrite the query — logs and responses should show
    # exactly what the model wrote (modulo whitespace/semicolon trimming).
    sql = "SELECT c.ticker AS t FROM companies c WHERE c.ticker = 'AAPL'"
    assert validate_sql(sql) == sql


# ---------------------------------------------------------------------------
# Rejected queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ;  ",
        "this is not sql at all (((",
    ],
)
def test_rejects_empty_or_unparseable(sql: str) -> None:
    with pytest.raises(SQLGuardError):
        validate_sql(sql)


def test_rejects_multiple_statements() -> None:
    # The classic chaining attack: a valid SELECT followed by a DROP.
    with pytest.raises(SQLGuardError) as exc_info:
        validate_sql("SELECT 1; DROP TABLE companies")
    assert "one" in exc_info.value.reason.lower()


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO companies (ticker) VALUES ('EVIL')",
        "UPDATE financial_metrics SET value = 0",
        "DELETE FROM filings",
        "DROP TABLE companies",
        "CREATE TABLE evil (id INT)",
        "ALTER TABLE companies ADD COLUMN evil TEXT",
        "TRUNCATE companies",
        "GRANT ALL ON companies TO PUBLIC",
    ],
)
def test_rejects_dml_and_ddl_at_top_level(sql: str) -> None:
    with pytest.raises(SQLGuardError):
        validate_sql(sql)


def test_rejects_data_modifying_cte() -> None:
    # Postgres allows WITH x AS (DELETE ... RETURNING *) SELECT ... — the top
    # level is a SELECT, so only a full-tree walk catches the DELETE inside.
    with pytest.raises(SQLGuardError):
        validate_sql(
            "WITH gone AS (DELETE FROM companies RETURNING id) SELECT * FROM gone"
        )


def test_rejects_select_into() -> None:
    # SELECT ... INTO is CREATE TABLE in a trench coat.
    with pytest.raises(SQLGuardError):
        validate_sql("SELECT * INTO evil_copy FROM companies")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT table_name FROM information_schema.tables",
        "SELECT * FROM pg_stat_activity",
    ],
)
def test_rejects_system_catalog_access(sql: str) -> None:
    # Schema snooping: the model should learn the schema from retrieved
    # context, not by introspecting the catalog.
    with pytest.raises(SQLGuardError):
        validate_sql(sql)


def test_rejects_set_statement() -> None:
    # SET could undo the statement_timeout that db.py installs.
    with pytest.raises(SQLGuardError):
        validate_sql("SET statement_timeout = 0")


def test_error_reason_is_instructive() -> None:
    # The reason string is fed back to the LLM as retry feedback — it must
    # say what is allowed, not just that something was rejected.
    with pytest.raises(SQLGuardError) as exc_info:
        validate_sql("DELETE FROM companies")
    reason = exc_info.value.reason
    assert isinstance(reason, str) and len(reason) > 20
    assert "SELECT" in reason
