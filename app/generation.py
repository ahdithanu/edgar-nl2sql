"""SQL generation and answer synthesis via the Anthropic API.

Two LLM responsibilities live here, kept deliberately separate from the
pipeline's control flow (app/pipeline.py owns *when* to call these; this
module owns *how* to talk to Claude):

- generate_sql: turn (question + retrieved context + prior failed attempts)
  into a single PostgreSQL SELECT. On retries the prompt includes every
  prior attempt's SQL and its failure, and asks Claude to DIAGNOSE the
  failure before regenerating — the diagnosis is captured and surfaced as
  SQLAttempt.correction_reasoning so a reviewer can watch the model reason
  its way to a fix across attempts.

- synthesize_answer: turn (question + SQL + result rows) into a short
  plain-English answer grounded ONLY in the rows we actually fetched.

Defensive parsing: LLMs are probabilistic text generators, not APIs. We ask
for SQL in a ```sql fenced block, but we tolerate a bare ``` fence and even
raw un-fenced SQL, because a retry loop that dies on a formatting quirk
wastes an attempt (and real money) on a non-error.
"""

from __future__ import annotations

import json
import re

import anthropic

from app.config import get_settings
from app.logging_config import get_logger
from app.models import ContextDoc, SQLAttempt

logger = get_logger(__name__)

# Lazily-constructed shared client: import must not require an API key
# (unit tests mock this module boundary), and the client reuses HTTP
# connections across requests once created.
_anthropic_client: anthropic.Anthropic | None = None

# max_tokens sizing: single-query SQL plus a short diagnosis paragraph fits
# comfortably; a hard ceiling also caps cost if the model rambles.
_GENERATION_MAX_TOKENS = 1500
_SYNTHESIS_MAX_TOKENS = 500

# Truncate huge error strings / row payloads before they hit the prompt —
# a psycopg traceback or a 200-row result set can drown the signal.
_MAX_ERROR_CHARS = 500
_MAX_ROWS_IN_PROMPT = 50

# The model's refusal protocol (see _SYSTEM_PROMPT). Matched before any SQL
# extraction so a refusal can never be parsed as a query.
_CANNOT_ANSWER_RE = re.compile(r"^\s*CANNOT_ANSWER:\s*(.+)$", re.MULTILINE)


class UnanswerableQuestionError(Exception):
    """The model reported the question cannot be answered from this schema.

    WHY this is its own exception rather than a generic failure: it is a
    *conclusion*, not an error. Retrying wastes LLM calls and — as the eval
    caught — actively pressures the model into manufacturing a fake row
    (e.g. `SELECT '<prose>' AS answer`) just to satisfy the loop. The
    pipeline short-circuits on this and reports an honest failure instead.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

_SQL_FENCE_RE = re.compile(r"```sql\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```\s*(.+?)```", re.DOTALL)

_SYSTEM_PROMPT = """You are an expert PostgreSQL analyst for a SEC EDGAR financial database.

Rules:
- Output exactly ONE PostgreSQL SELECT statement (WITH/CTEs are fine if the final statement is a SELECT).
- Never write INSERT/UPDATE/DELETE/DDL of any kind. Never query pg_catalog or information_schema.
- Use ONLY the tables, columns, and semantics described in the provided context documents. Do not invent tables or columns.
- Match filters to the exact stored values described in the context (e.g. metric names like 'net_income', fiscal_period values like 'FY' or 'Q2', tickers are uppercase).
- Prefer explicit JOINs and readable formatting. Alias aggregate/computed columns clearly.
- Put the SQL in a ```sql fenced code block.
- If the question CANNOT be answered from the described schema (it asks about data this
  database does not hold), do NOT invent a query and do NOT return prose inside a SELECT
  literal. Reply with exactly one line, no code block:
  CANNOT_ANSWER: <one sentence explaining what data would be needed and why it is absent>"""


def _get_anthropic_client() -> anthropic.Anthropic:
    """Return the shared Anthropic client, creating it on first use."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(
            api_key=get_settings().anthropic_api_key
        )
    return _anthropic_client


def _call_claude(prompt: str, max_tokens: int) -> str:
    """Single-turn Claude call; returns the concatenated text content."""
    settings = get_settings()
    response = _get_anthropic_client().messages.create(
        model=settings.claude_model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    # A message can contain multiple content blocks; keep only text ones.
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


def _extract_sql(text: str) -> str:
    """Pull the SQL statement out of a model response, defensively.

    Preference order (strictest first):
    1. ```sql fenced block — what we explicitly asked for.
    2. Any ``` fenced block — models sometimes drop the language tag.
    3. Raw text from the first SELECT/WITH keyword — last resort for a
       model that ignored fencing entirely.
    Raises ValueError if no SQL-shaped content is found, which the pipeline
    treats as a failed attempt (never a crash).

    Raises UnanswerableQuestionError when the model used the CANNOT_ANSWER
    protocol — checked FIRST, because that reply is a deliberate refusal and
    must never be retried or mistaken for a malformed response.
    """
    refusal = _CANNOT_ANSWER_RE.search(text)
    if refusal:
        raise UnanswerableQuestionError(refusal.group(1).strip())

    match = _SQL_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    match = _ANY_FENCE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        # The any-fence regex captures the fence's language tag along with the
        # body (```postgresql, ```pgsql, ...). If the first line is a single
        # bare word that isn't the start of the SQL itself, drop it.
        first_line, sep, rest = candidate.partition("\n")
        if (
            sep
            and re.fullmatch(r"[A-Za-z][\w+-]*", first_line.strip())
            and first_line.strip().upper() not in ("SELECT", "WITH")
        ):
            candidate = rest.strip()
        if candidate:
            return candidate

    # No fences at all: look for a bare SELECT / WITH and take the rest.
    bare = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if bare:
        return text[bare.start():].strip()

    raise ValueError("model response contained no recognizable SQL")


def _format_context(context: list[ContextDoc]) -> str:
    """Render retrieved docs verbatim, tagged with type and title.

    Verbatim injection is intentional: the corpus docs were written to be
    read by the model as-is, and any summarization here would risk losing
    the exact stored values (metric names, fiscal_period codes) that the
    SQL must match literally.
    """
    if not context:
        return "(no context documents retrieved)"
    return "\n\n".join(
        f"[{doc.doc_type}] {doc.title}\n{doc.content}" for doc in context
    )


def _format_prior_attempts(prior_attempts: list[SQLAttempt]) -> str:
    """Render each failed attempt: the SQL, how it failed, and the error.

    This is the feedback channel that makes the retry loop *agentic* rather
    than a blind re-roll: attempt N sees exactly what attempts 1..N−1 tried
    and why each one failed, so it can correct the specific mistake instead
    of resampling the same one.
    """
    sections: list[str] = []
    for attempt in prior_attempts:
        error = (attempt.error_message or "(no error message)")[:_MAX_ERROR_CHARS]
        sections.append(
            f"--- Attempt {attempt.attempt_number} ---\n"
            f"SQL:\n{attempt.sql}\n"
            f"Outcome: {attempt.outcome}\n"
            f"Error: {error}"
        )
    return "\n\n".join(sections)


def generate_sql(
    question: str,
    context: list[ContextDoc],
    prior_attempts: list[SQLAttempt],
) -> tuple[str, str | None]:
    """Generate a SQL query for the question; returns (sql, correction_reasoning).

    First attempt: context + question → SQL. correction_reasoning is None.

    Retry attempts: the prompt additionally contains every prior attempt's
    SQL, outcome, and error, and instructs Claude to diagnose the failure
    FIRST, then write corrected SQL. Diagnosis-before-regeneration matters:
    forcing the model to articulate *why* the last query failed measurably
    reduces the chance it repeats the same class of mistake, and the text
    it produces becomes correction_reasoning — an audit trail of the
    model's self-correction, returned to the caller in each SQLAttempt.
    """
    context_block = _format_context(context)

    if not prior_attempts:
        prompt = (
            f"Context documents (schema, column semantics, glossary):\n\n"
            f"{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Write the SQL query that answers this question."
        )
    else:
        prompt = (
            f"Context documents (schema, column semantics, glossary):\n\n"
            f"{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Previous attempts at this question FAILED:\n\n"
            f"{_format_prior_attempts(prior_attempts)}\n\n"
            f"First, in one short paragraph, diagnose why the most recent "
            f"attempt failed (wrong table? wrong stored value? bad join? "
            f"forbidden statement?). Then write a corrected SQL query in a "
            f"```sql fenced block. Do not repeat a query that already failed."
        )

    text = _call_claude(prompt, _GENERATION_MAX_TOKENS)
    sql = _extract_sql(text)

    correction_reasoning: str | None = None
    if prior_attempts:
        # The diagnosis is whatever prose precedes the SQL fence. Fall back
        # to the full response (minus the SQL) if the fence came first.
        fence = _SQL_FENCE_RE.search(text) or _ANY_FENCE_RE.search(text)
        prose = text[: fence.start()].strip() if fence else text.replace(sql, "").strip()
        correction_reasoning = prose or None

    return sql, correction_reasoning


def synthesize_answer(question: str, sql: str, rows: list[dict]) -> str:
    """Produce a short plain-English answer grounded ONLY in the rows.

    Grounding rule: the model may not "know" answers — the rows ARE the
    answer, the model just narrates them. This is what keeps the final
    answer trustworthy: every number in it is traceable to a returned row,
    and the SQL + rows are in the response for anyone who wants to verify.
    """
    settings = get_settings()

    truncation_note = ""
    if len(rows) >= settings.max_result_rows:
        # execute_readonly caps fetches at max_result_rows, so hitting the
        # cap means the true result set may be larger — say so honestly.
        truncation_note = (
            f"\nNOTE: the result set was truncated at {settings.max_result_rows} "
            f"rows; tell the user the list may be incomplete."
        )

    # Cap rows in the prompt: 50 rows is plenty for narration, and keeps
    # token cost bounded. default=str catches any type json can't handle.
    shown_rows = rows[:_MAX_ROWS_IN_PROMPT]
    rows_json = json.dumps(shown_rows, default=str)
    if len(rows) > _MAX_ROWS_IN_PROMPT:
        rows_json += f"\n... ({len(rows) - _MAX_ROWS_IN_PROMPT} more rows not shown)"

    prompt = (
        f"Question: {question}\n\n"
        f"SQL that was executed:\n{sql}\n\n"
        f"Result rows (JSON):\n{rows_json}\n"
        f"{truncation_note}\n\n"
        f"Answer the question in 1-3 plain-English sentences using ONLY the "
        f"result rows above. State the key figures with sensible formatting "
        f"(e.g. $383.3 billion). Do not use any outside knowledge, and do "
        f"not mention SQL or databases."
    )

    return _call_claude(prompt, _SYNTHESIS_MAX_TOKENS).strip()
