"""Context retrieval: pgvector similarity search over the RAG corpus.

============================================================================
RETRIEVAL HAPPENS BEFORE GENERATION — this is the "R" in RAG, and it is the
first thing the pipeline does on every request, *before* Claude is asked to
write a single character of SQL.

WHY this ordering matters:

1. Claude has never seen this database. Its training data contains millions
   of SQL examples but zero rows of *our* schema. Without retrieved context
   it will hallucinate plausible-looking table names ("revenues",
   "quarterly_financials") that simply do not exist here.

2. The schema alone is not enough — the *semantics* are the hard part.
   Example: `total_assets` has no Q4 row (it's a balance-sheet snapshot;
   the FY row IS the year-end value), while `revenue` Q4 is derived as
   FY − (Q1+Q2+Q3). A model that doesn't know this writes syntactically
   valid SQL that returns silently wrong numbers. Retrieval injects these
   footguns as glossary documents exactly when they're relevant.

3. It keeps the prompt small and targeted. We could paste the entire corpus
   into every prompt, but retrieval selects the top-k most relevant docs
   for *this* question — cheaper, and the model attends better to a short,
   relevant context than to a wall of boilerplate.

4. Transparency: the retrieved docs are returned to the caller in
   QueryResponse.context_docs, so a reviewer can see exactly what grounding
   the model was given for any answer. If generation goes wrong, the first
   debugging question is "what did retrieval feed it?" — and the answer is
   right there in the response.

Mechanics: questions are embedded with Voyage (input_type="query" — Voyage
embeds queries and documents into slightly different subspaces so that
questions land near the documents that answer them), then matched against
pre-embedded corpus docs in Postgres via pgvector's cosine-distance
operator `<=>`. One database holds both the relational data and the
embeddings, so there is no separate vector store to deploy or drift.
============================================================================
"""

from __future__ import annotations

import time

import voyageai

from app.config import get_settings
from app.db import get_pool
from app.logging_config import get_logger
from app.models import ContextDoc

logger = get_logger(__name__)

# Bounded backoff for embedding-provider rate limits. Voyage's free tier
# without a payment method allows only 3 requests/minute, and even paid tiers
# can 429 under bursts — one embed call per query means a single 429 would
# otherwise kill the whole request before generation ever ran. Three waits
# capped at ~75s total keeps us inside a 3 RPM window without retrying forever.
_EMBED_RATE_LIMIT_RETRIES = 3
_EMBED_BACKOFF_SECONDS = (5.0, 25.0, 45.0)

# Lazily-constructed module-level client. Lazy because importing this module
# must not require a VOYAGE_API_KEY (unit tests import and mock it); module
# level because the client holds an HTTP connection pool worth reusing.
_voyage_client: voyageai.Client | None = None


def _get_voyage_client() -> voyageai.Client:
    """Return the shared Voyage client, creating it on first use."""
    global _voyage_client
    if _voyage_client is None:
        _voyage_client = voyageai.Client(api_key=get_settings().voyage_api_key)
    return _voyage_client


def embed_query(text: str) -> list[float]:
    """Embed a user question into the same 1024-dim space as the corpus.

    input_type="query" is deliberate and load-bearing: the corpus was
    embedded with input_type="document" (see scripts/build_embeddings.py),
    and Voyage's asymmetric query/document encoding measurably improves
    question→document matching versus embedding both sides identically.
    """
    settings = get_settings()
    for attempt in range(_EMBED_RATE_LIMIT_RETRIES + 1):
        try:
            result = _get_voyage_client().embed(
                [text],
                model=settings.embed_model,
                input_type="query",
            )
            return result.embeddings[0]
        except Exception as exc:
            # Duck-typed rate-limit detection: voyageai raises RateLimitError,
            # but matching on class name + message keeps this robust across
            # SDK versions without importing private exception hierarchies.
            is_rate_limit = "ratelimit" in type(exc).__name__.lower() or "rate limit" in str(exc).lower()
            if not is_rate_limit or attempt == _EMBED_RATE_LIMIT_RETRIES:
                raise
            delay = _EMBED_BACKOFF_SECONDS[attempt]
            logger.warning(
                "embed_rate_limited",
                attempt=attempt + 1,
                retry_in_s=delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop always returns or raises


def retrieve_context(question: str, k: int | None = None) -> list[ContextDoc]:
    """Return the k most relevant context docs for a question.

    Similarity search runs entirely inside Postgres: `<=>` is pgvector's
    cosine-distance operator, accelerated by the HNSW index on
    rag_documents.embedding. We convert distance to similarity
    (similarity = 1 − cosine_distance) because "higher is better" is the
    intuitive reading for anyone inspecting context_docs in a response.
    """
    settings = get_settings()
    top_k = k if k is not None else settings.retrieval_top_k

    embedding = embed_query(question)
    # pgvector's text input format: '[0.1,0.2,...]'. Passed as a parameter
    # and cast server-side (%s::vector) — never interpolated into the SQL.
    vector_param = "[" + ",".join(str(v) for v in embedding) + "]"

    # Plain parameterized query on the shared pool. This is our own trusted
    # SQL, not model output, so it does not go through the sql_guard /
    # read-only execution path that model-generated SQL must pass.
    query = """
        SELECT doc_type,
               title,
               content,
               1 - (embedding <=> %s::vector) AS similarity
        FROM rag_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (vector_param, vector_param, top_k))
            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()

    # The pool's row factory is db.py's choice (tuple vs dict rows); zip
    # against cursor.description so this works either way.
    docs = []
    for row in rows:
        record = row if isinstance(row, dict) else dict(zip(columns, row))
        docs.append(
            ContextDoc(
                doc_type=record["doc_type"],
                title=record["title"],
                content=record["content"],
                similarity=float(record["similarity"]),
            )
        )

    logger.info(
        "context_retrieved",
        question=question,
        k=top_k,
        doc_titles=[d.title for d in docs],
        top_similarity=docs[0].similarity if docs else None,
    )
    return docs
