"""Static validation gate for LLM-generated SQL.

WHY THIS MODULE EXISTS
----------------------
The SQL executed by this service is written by a language model, not a human.
Even with a well-crafted prompt, we must assume the model can emit anything:
mutations, schema snooping, multiple statements, or plain garbage. This module
is the first of two independent safety layers:

1. **This guard (static analysis)** — parse the SQL with sqlglot's Postgres
   dialect and reject anything that is not a single, read-only SELECT before
   it ever touches the database.
2. **Read-only transaction (runtime)** — ``app.db.execute_readonly`` runs every
   query inside ``SET TRANSACTION READ ONLY`` with a statement timeout, so even
   a guard bypass cannot mutate data (defense in depth).

DESIGN NOTE: error messages are part of the product. When the guard rejects a
query, ``SQLGuardError.reason`` is fed *back to the LLM* as the error for the
next self-correction attempt (see ``app.pipeline``). Vague messages produce
vague retries, so every rejection explains what was wrong and what is allowed.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class SQLGuardError(Exception):
    """Raised when generated SQL fails validation.

    ``reason`` is a human- (and LLM-) readable explanation. The pipeline
    includes it verbatim in the retry prompt, so it should tell the model
    how to fix the problem, not just that one exists.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


# Statement node types that are never acceptable ANYWHERE in the tree — not
# just at the top level. Postgres allows data-modifying CTEs like
# ``WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x``, so a top-level
# check alone is insufficient; we walk every node.
#
# Names are resolved via getattr because sqlglot has renamed some classes
# across versions (e.g. AlterTable -> Alter). Missing names are skipped
# rather than crashing on import.
_FORBIDDEN_NODE_NAMES: tuple[str, ...] = (
    # DML
    "Insert",
    "Update",
    "Delete",
    "Merge",
    # DDL
    "Create",
    "Drop",
    "Alter",
    "AlterTable",
    "TruncateTable",
    # Permissions / session / execution
    "Grant",
    "Revoke",
    "Copy",
    "Set",          # SET <param> — session tampering (e.g. resetting timeouts)
    "Use",
    "Transaction",  # BEGIN / START TRANSACTION
    "Commit",
    "Rollback",
    # sqlglot parses statements it has no dedicated node for (CALL, EXECUTE,
    # VACUUM, DO, ...) into a generic Command node — reject those wholesale.
    "Command",
)

_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    cls
    for name in _FORBIDDEN_NODE_NAMES
    if (cls := getattr(exp, name, None)) is not None
)

# System schemas/prefixes the LLM has no business querying. Blocking these
# prevents "schema snooping" — the model should learn the schema from the
# retrieved RAG context, not by introspecting the catalog (which would also
# leak infrastructure details in a multi-tenant database like Supabase).
_BLOCKED_SCHEMAS = frozenset({"pg_catalog", "information_schema"})
_BLOCKED_TABLE_PREFIX = "pg_"

# Dangerous function calls. Node-type and table checks don't see these: a
# plain SELECT can still call functions that execute SQL on a SEPARATE
# connection (dblink — outside our READ ONLY transaction!), run arbitrary
# queries (query_to_xml and the *_to_xml family), read server-local files
# (pg_read_file, lo_import), tamper with session state (set_config), or waste
# pool connections (pg_sleep). The pg_/dblink prefixes cover the admin and
# introspection families wholesale (pg_sleep, pg_read_file, pg_ls_dir,
# pg_terminate_backend, dblink_exec, dblink_connect, ...).
_BLOCKED_FUNCTION_PREFIXES: tuple[str, ...] = ("pg_", "dblink")
_BLOCKED_FUNCTIONS = frozenset(
    {
        "lo_import",
        "lo_export",
        "lo_get",
        "lo_put",
        "loread",
        "lowrite",
        "query_to_xml",
        "query_to_xmlschema",
        "query_to_xml_and_xmlschema",
        "database_to_xml",
        "database_to_xmlschema",
        "database_to_xml_and_xmlschema",
        "schema_to_xml",
        "schema_to_xmlschema",
        "schema_to_xml_and_xmlschema",
        "table_to_xml",
        "table_to_xmlschema",
        "table_to_xml_and_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
        "set_config",
        "current_setting",
        "txid_current",
    }
)


def _function_name(node: exp.Expression) -> str | None:
    """Return the lowercased call name for a function node, else None.

    sqlglot parses functions it knows into dedicated exp.Func subclasses
    (SUM -> exp.Sum) and everything else (pg_sleep, dblink, ...) into
    exp.Anonymous, whose .name holds the raw call name.
    """
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    if isinstance(node, exp.Func):
        return node.sql_name().lower()
    return None

_ALLOWED_HINT = (
    "Only a single read-only SELECT statement over the application tables "
    "(companies, filings, financial_metrics) is allowed."
)


def validate_sql(sql: str) -> str:
    """Validate LLM-generated SQL and return it normalized, or raise SQLGuardError.

    Checks, in order (first failure wins):
      1. Non-empty and parseable as Postgres SQL.
      2. Exactly one statement (blocks classic ``...; DROP TABLE`` chaining).
      3. Top-level expression is a SELECT. WITH/CTEs are fine as long as the
         final expression is a SELECT; set operations (UNION/INTERSECT/EXCEPT)
         of SELECTs are also accepted since they are read-only by construction.
      4. No DML/DDL/session-control node anywhere in the tree (catches
         data-modifying CTEs and subquery tricks, not just top-level verbs).
      5. No ``SELECT ... INTO`` (a sneaky way to CREATE TABLE from a SELECT).
      6. No ``FOR UPDATE/SHARE`` row locking (pointless in a read-only query,
         and it would only die later inside the READ ONLY transaction).
      7. No dangerous function calls (dblink/pg_*/lo_*/query_to_xml/... —
         these can execute SQL outside the READ ONLY transaction, read server
         files, or tie up connections even from inside a SELECT).
      8. No access to pg_catalog / information_schema / pg_* tables.

    Normalization is deliberately minimal: strip surrounding whitespace and
    trailing semicolons. We return the model's own SQL text (not sqlglot's
    re-rendering) so that what we log, execute, and show the user is exactly
    what the model wrote.
    """
    normalized = sql.strip().rstrip(";").strip()

    if not normalized:
        raise SQLGuardError(
            "The SQL statement is empty. Respond with one SELECT statement in a "
            "```sql fenced block."
        )

    # --- 1. Parse (Postgres dialect, matching the execution target) --------
    try:
        statements = [
            s for s in sqlglot.parse(normalized, dialect="postgres") if s is not None
        ]
    except ParseError as e:
        # Surface sqlglot's diagnostic — it usually pinpoints the token that
        # broke, which is exactly what the model needs to fix its syntax.
        raise SQLGuardError(
            f"The SQL failed to parse as PostgreSQL: {e}. "
            "Emit syntactically valid PostgreSQL."
        ) from e

    if not statements:
        raise SQLGuardError(
            "No SQL statement was found. Respond with one SELECT statement in a "
            "```sql fenced block."
        )

    # --- 2. Single statement only ------------------------------------------
    if len(statements) > 1:
        raise SQLGuardError(
            f"Found {len(statements)} SQL statements; exactly one is allowed. "
            "Combine the logic into a single SELECT (CTEs/WITH are permitted)."
        )

    tree = statements[0]

    # --- 3. Top level must be a SELECT --------------------------------------
    # sqlglot attaches a WITH clause to its final expression, so a valid
    # ``WITH ... SELECT`` parses as a Select node here — no special casing
    # needed. exp.Union is the base class for all set operations in sqlglot
    # (INTERSECT and EXCEPT subclass it), and their operands are themselves
    # validated by the node walk below.
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise SQLGuardError(
            f"Top-level statement is {type(tree).__name__.upper()}, not SELECT. "
            + _ALLOWED_HINT
        )

    # --- 4. No forbidden node anywhere in the tree ---------------------------
    # This is the workhorse check: it walks EVERY node, so an INSERT hiding
    # inside a CTE or subquery is caught even though the outer statement is a
    # SELECT.
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            verb = type(node).__name__.upper()
            raise SQLGuardError(
                f"Statement contains a forbidden {verb} operation. "
                + _ALLOWED_HINT
            )

    # --- 5. No SELECT ... INTO ----------------------------------------------
    if tree.find(exp.Into) is not None:
        raise SQLGuardError(
            "SELECT ... INTO creates a table and is not allowed. "
            "Return the rows directly from a plain SELECT."
        )

    # --- 6. No row locking ----------------------------------------------------
    if tree.find(exp.Lock) is not None:
        raise SQLGuardError(
            "FOR UPDATE/FOR SHARE row locking is not allowed in a read-only "
            "query. Remove the locking clause."
        )

    # --- 7. No dangerous function calls ---------------------------------------
    # A read-only SELECT can still do damage through functions: dblink() runs
    # SQL on a NEW connection outside our READ ONLY transaction, query_to_xml
    # executes arbitrary SQL, pg_read_file/lo_import touch the server
    # filesystem, and pg_sleep ties up pooled connections. Block them here.
    for node in tree.walk():
        fname = _function_name(node)
        if fname is None:
            continue
        if fname in _BLOCKED_FUNCTIONS or fname.startswith(_BLOCKED_FUNCTION_PREFIXES):
            raise SQLGuardError(
                f"Call to function '{fname}' is not allowed. Use only ordinary "
                "SQL expressions and aggregates over the application tables. "
                + _ALLOWED_HINT
            )

    # --- 8. No system-catalog access -----------------------------------------
    for table in tree.find_all(exp.Table):
        schema = (table.db or "").lower()
        catalog = (table.catalog or "").lower()
        name = table.name.lower()
        if (
            schema in _BLOCKED_SCHEMAS
            or catalog in _BLOCKED_SCHEMAS
            or name.startswith(_BLOCKED_TABLE_PREFIX)
        ):
            raise SQLGuardError(
                f"Access to system table '{table.sql(dialect='postgres')}' is not "
                "allowed. Query only the application tables described in the "
                "provided schema context (companies, filings, financial_metrics)."
            )

    return normalized
