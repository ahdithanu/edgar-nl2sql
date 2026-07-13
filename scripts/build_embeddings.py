"""Embed the RAG corpus and upsert it into rag_documents.

Run AFTER scripts/schema.sql has been applied:

    python scripts/build_embeddings.py

What it does:
1. Loads CONTEXT_DOCS from scripts/context_docs.py (the corpus source of truth).
2. Embeds every doc with Voyage — model from Settings.embed_model
   (voyage-3.5-lite, 1024 dims), input_type="document". The query side
   (app/retrieval.py) embeds with input_type="query"; Voyage's asymmetric
   encoding is why the two sides must NOT be swapped.
3. Upserts into rag_documents keyed on title: delete-then-insert per run, so
   editing a doc's content in context_docs.py and rerunning this script is
   the entire update workflow (no drift between file and database).

Idempotent and safe to rerun at any time.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runnable as `python scripts/build_embeddings.py` from the repo root: put the
# repo root on sys.path so `app.*` and the sibling context_docs import resolve.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import voyageai  # noqa: E402

from app.config import get_settings  # noqa: E402
from scripts.context_docs import CONTEXT_DOCS  # noqa: E402

# Voyage accepts up to 128 inputs per embed call; batching keeps us to a
# handful of round trips even if the corpus grows.
BATCH_SIZE = 128


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus texts with input_type='document', batched."""
    settings = get_settings()
    client = voyageai.Client(api_key=settings.voyage_api_key)
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        result = client.embed(
            batch,
            model=settings.embed_model,
            input_type="document",
        )
        embeddings.extend(result.embeddings)
    return embeddings


def main() -> int:
    docs = CONTEXT_DOCS
    if not docs:
        print("context_docs.CONTEXT_DOCS is empty — nothing to embed.")
        return 1

    print(f"Embedding {len(docs)} context documents ...")
    embeddings = embed_documents([d["content"] for d in docs])
    if len(embeddings) != len(docs):
        print(
            f"ERROR: got {len(embeddings)} embeddings for {len(docs)} docs.",
            file=sys.stderr,
        )
        return 1

    # Import here (after embedding) so a Voyage failure surfaces before we
    # open any database connection.
    import psycopg

    settings = get_settings()
    upserted = 0
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            for doc, embedding in zip(docs, embeddings):
                vector_param = "[" + ",".join(str(v) for v in embedding) + "]"
                # Delete-then-insert keyed on title: simplest correct upsert
                # for a small corpus, and it guarantees stale content/doc_type
                # never survives a rerun.
                cur.execute(
                    "DELETE FROM rag_documents WHERE title = %s", (doc["title"],)
                )
                cur.execute(
                    """
                    INSERT INTO rag_documents (doc_type, title, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    """,
                    (doc["doc_type"], doc["title"], doc["content"], vector_param),
                )
                upserted += 1
        conn.commit()

    print(f"Upserted {upserted} documents into rag_documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
