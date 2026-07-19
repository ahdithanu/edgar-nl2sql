"""Pydantic models shared across the API, pipeline, and eval harness.

These are the wire types: `QueryRequest` in, `QueryResponse` out. The response
deliberately exposes the system's full working — every SQL attempt (including failures)
and every retrieved context document — because for a demo/portfolio system the
*transparency* of the RAG + retry machinery is the product, not just the final answer.
"""

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A natural-language financial question.

    Bounds are enforced here so garbage (empty strings, novel-length prompts) is
    rejected with a 422 before we spend money on embeddings or LLM calls.
    """

    question: str = Field(min_length=3, max_length=1000)


class ContextDoc(BaseModel):
    """One document retrieved from the RAG corpus for a given question.

    Returned to the caller verbatim so users (and hiring managers reading demo output)
    can see exactly what schema/glossary knowledge the model was grounded in.
    """

    doc_type: str  # 'table_schema' | 'column' | 'glossary'
    title: str
    content: str
    similarity: float  # 1 - cosine distance; higher = more relevant


class SQLAttempt(BaseModel):
    """A single iteration of the generate -> validate -> execute loop.

    The pipeline records one of these per attempt, success or failure. Keeping the
    failures (with the model's own `correction_reasoning`) makes the self-correction
    loop auditable: you can read exactly why attempt 2 differs from attempt 1.
    """

    attempt_number: int  # 1-based
    sql: str
    # "unanswerable" is terminal, not a retry signal: the model reported the
    # schema cannot answer the question, so the pipeline stops immediately
    # rather than spending further attempts (see app/pipeline.py).
    outcome: Literal[
        "success", "guard_rejected", "execution_error", "empty_result", "unanswerable"
    ]
    error_message: str | None = None
    # The model's diagnosis of the previous failure that produced THIS attempt.
    # Always None on attempt 1 (there was nothing to correct yet).
    correction_reasoning: str | None = None
    duration_ms: int


class QueryResponse(BaseModel):
    """The full result of running the pipeline on one question."""

    request_id: str
    question: str
    success: bool
    # Final successful SQL. None when all attempts failed — `attempts` still holds
    # every SQL string that was tried, so nothing is hidden.
    sql: str | None
    rows: list[dict] = []
    # Plain-English answer grounded in `rows`, OR (on failure) a clear explanation of
    # what was tried and why it didn't work. Never empty.
    answer: str
    attempts: list[SQLAttempt]
    context_docs: list[ContextDoc] = []  # what retrieval injected (transparency/demo value)


class HealthResponse(BaseModel):
    """Liveness/readiness snapshot for GET /health."""

    status: Literal["ok", "degraded"]
    database: bool
    version: str = "0.1.0"
