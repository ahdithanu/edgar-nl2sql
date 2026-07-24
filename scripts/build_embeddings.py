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


def build_coverage_doc() -> dict:
    """Generate a 'data coverage' doc by querying what is actually loaded.

    WHY this is generated rather than written by hand: the corpus used to
    assert "fiscal 2020 through 2024" in prose. When the loader's year window
    was widened, that sentence silently became a lie, and because the model
    only knows what the corpus tells it, it began refusing questions about
    years that were sitting right there in the table. Coverage claims must be
    derived from the data or they drift.
    """
    import psycopg

    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT sic_description) FROM companies")
        n_companies, n_industries = cur.fetchone()
        cur.execute(
            "SELECT min(fiscal_year), max(fiscal_year), count(*) FROM financial_metrics"
        )
        yr_min, yr_max, n_rows = cur.fetchone()
        # The most recent year is usually partial: report which companies have
        # a full-year row there so "latest annual" questions resolve correctly.
        cur.execute(
            "SELECT count(DISTINCT company_id) FROM financial_metrics "
            "WHERE fiscal_year = %s AND fiscal_period = 'FY'",
            (yr_max,),
        )
        (n_with_fy_latest,) = cur.fetchone()
        # "Broad coverage" is measured against the PEAK annual FY count, not
        # the total company count. The universe grew over time (recent IPOs,
        # newer filers), so no single year has an FY row for every company;
        # comparing to total companies would wrongly reject even the best year
        # and then fall through to the current partial year. Peak-relative
        # correctly flags the newest partial year while keeping fully-filed
        # years in scope.
        cur.execute(
            "SELECT year, n FROM ("
            "  SELECT fiscal_year AS year, count(DISTINCT company_id) AS n "
            "  FROM financial_metrics WHERE fiscal_period = 'FY' GROUP BY fiscal_year"
            ") t ORDER BY n DESC LIMIT 1"
        )
        row = cur.fetchone()
        peak_fy = row[1] if row else 0
        cur.execute(
            "SELECT max(fiscal_year) FROM financial_metrics f WHERE fiscal_period = 'FY' "
            "AND (SELECT count(DISTINCT company_id) FROM financial_metrics x "
            "     WHERE x.fiscal_year = f.fiscal_year AND x.fiscal_period = 'FY') >= %s",
            (peak_fy * 0.5,),
        )
        (last_broad_fy,) = cur.fetchone()
        if last_broad_fy is None:  # degenerate (no FY rows at all)
            last_broad_fy = yr_max

    content = f"""
    Data coverage, generated from the database at embedding time.

    - Companies: {n_companies}, spanning {n_industries} SIC industry
      classifications. These are the largest US-listed filers by market
      capitalization, per the ordering SEC publishes in company_tickers.json.
    - Fiscal years loaded: {yr_min} through {yr_max}.
    - Rows in financial_metrics: {n_rows}.
    - Fiscal year {yr_max} is PARTIAL: only {n_with_fy_latest} of {n_companies}
      companies have a full-year ('FY') row for it, because many fiscal years
      have not ended or been filed yet.
    - The most recent fiscal year with broad full-year coverage is {last_broad_fy}.

    Consequences for query writing:
    - "latest" / "most recent" annual figures: prefer the latest year that has
      fiscal_period = 'FY' FOR THAT COMPANY, e.g.
        ... WHERE fm.metric='revenue' AND fm.fiscal_period='FY'
            AND c.ticker='WMT' ORDER BY fm.fiscal_year DESC LIMIT 1
      Do not assume every company has an FY row for {yr_max}.
    - Cross-company comparisons and "which company had the highest X" should
      normally use {last_broad_fy} or earlier, otherwise the ranking silently
      covers only the subset that has already filed.
    - A question about a year outside {yr_min}-{yr_max} is not answerable from
      this database. Say so rather than substituting a nearby year.
    """
    return _doc_dict("glossary", "data coverage — companies, years, and partial-year caveat", content)


def _doc_dict(doc_type: str, title: str, content: str) -> dict:
    from textwrap import dedent

    return {"doc_type": doc_type, "title": title, "content": dedent(content).strip()}


def main() -> int:
    docs = list(CONTEXT_DOCS)
    if not docs:
        print("context_docs.CONTEXT_DOCS is empty — nothing to embed.")
        return 1

    # Appended last so it is embedded alongside the hand-written docs.
    try:
        docs.append(build_coverage_doc())
        print("Generated data-coverage doc from the live database.")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not generate coverage doc: {exc}", file=sys.stderr)
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
