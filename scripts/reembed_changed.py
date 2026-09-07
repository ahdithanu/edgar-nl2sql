"""Re-embed only the corpus documents whose text changed.

`build_embeddings.py` rebuilds the whole table, which re-embeds all 31
documents and re-pays for the 28 that did not change. When a correction touches
a few documents, this script embeds only those and leaves every other row --
and its embedding -- byte-identical.

A renamed document appears as one removal plus one addition, because rows are
matched by title. The net table still mirrors the corpus exactly.

Usage:
    python scripts/reembed_changed.py            # plan only, no writes
    python scripts/reembed_changed.py --apply    # embed and write
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from scripts.build_embeddings import build_coverage_doc, embed_documents  # noqa: E402
from scripts.context_docs import CONTEXT_DOCS  # noqa: E402


def corpus_documents():
    """The corpus as build_embeddings.py assembles it, coverage doc included."""
    return list(CONTEXT_DOCS) + [build_coverage_doc()]


def plan(live, docs):
    by_title = {d["title"]: d for d in docs}
    changed = [d for t, d in by_title.items()
               if t in live and (live[t]["content"] != d["content"]
                                 or live[t]["doc_type"] != d["doc_type"])]
    added = [d for t, d in by_title.items() if t not in live]
    removed = [t for t in live if t not in by_title]
    return changed, added, removed


def main(argv):
    apply = "--apply" in argv
    import psycopg

    docs = corpus_documents()
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_type, title, content FROM rag_documents")
            live = {t: {"doc_type": dt, "content": c} for dt, t, c in cur.fetchall()}

        changed, added, removed = plan(live, docs)
        print(f"corpus {len(docs)} documents, live table {len(live)} rows")
        for doc in changed:
            print(f"  CHANGED  {doc['title']}")
        for doc in added:
            print(f"  ADDED    {doc['title']}")
        for title in removed:
            print(f"  REMOVED  {title}")
        if not (changed or added or removed):
            print("nothing to do; live table already matches the corpus.")
            return 0

        to_embed = changed + added
        print(f"\n{len(to_embed)} document(s) need embedding; "
              f"{len(docs) - len(to_embed)} left untouched.")
        if not apply:
            print("plan only — pass --apply to write.")
            return 0

        embeddings = embed_documents([d["content"] for d in to_embed]) if to_embed else []
        if len(embeddings) != len(to_embed):
            print(f"ERROR: got {len(embeddings)} embeddings for {len(to_embed)} docs.",
                  file=sys.stderr)
            return 1

        with conn.cursor() as cur:
            for title in removed:
                cur.execute("DELETE FROM rag_documents WHERE title = %s", (title,))
            for doc, embedding in zip(to_embed, embeddings):
                vector = "[" + ",".join(str(v) for v in embedding) + "]"
                cur.execute(
                    """
                    INSERT INTO rag_documents (doc_type, title, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (title) DO UPDATE
                       SET doc_type = EXCLUDED.doc_type,
                           content = EXCLUDED.content,
                           embedding = EXCLUDED.embedding
                    """,
                    (doc["doc_type"], doc["title"], doc["content"], vector),
                )
        conn.commit()
    print(f"\nwrote {len(to_embed)} embedded document(s), removed {len(removed)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
