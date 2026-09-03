"""Retrieval over 10-K narrative chunks (the document-RAG path).

Parallel to app.retrieval (which searches the schema/glossary corpus for the
SQL path), this searches filing_chunks: real Risk Factors and MD&A text pulled
from 10-K filings. Used to answer qualitative questions ("what risks did Apple
cite?", "how did management explain the change?") that no amount of SQL over
the metric table can answer.

Retrieval optionally filters to a single company. WHY: a question like "what
are Apple's biggest risks" should not surface Microsoft's risk factors just
because they embed similarly; when the question names a company we scope the
cosine search to that company's chunks, which sharply improves precision.
"""

from __future__ import annotations

from app.db import get_pool
from app.logging_config import get_logger
from app.models import ContextDoc
from app.retrieval import embed_query

logger = get_logger(__name__)

_SECTION_LABEL = {"risk_factors": "Risk Factors", "mda": "MD&A"}


def retrieve_filing_chunks(
    question: str, company_id: int | None = None, k: int = 6
) -> list[ContextDoc]:
    """The k most relevant 10-K chunks for a question, optionally one company.

    Returns ContextDoc so the API response shape is identical to the SQL
    path's context_docs: title is "TICKER 10-K <Section>", content is the
    chunk, similarity is 1 − cosine distance. doc_type is "filing:<section>"
    so a reader can tell filing sources from schema/glossary docs.
    """
    embedding = embed_query(question)
    vector_param = "[" + ",".join(str(v) for v in embedding) + "]"

    query = """
        SELECT c.ticker,
               fc.section,
               fc.accession,
               fc.fiscal_year,
               fc.content,
               1 - (fc.embedding <=> %s::vector) AS similarity
        FROM filing_chunks fc
        JOIN companies c ON c.id = fc.company_id
        {where}
        ORDER BY fc.embedding <=> %s::vector
        LIMIT %s
    """.format(where="WHERE fc.company_id = %s" if company_id is not None else "")

    params: list = [vector_param]
    if company_id is not None:
        params.append(company_id)
    params += [vector_param, k]

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()

    docs: list[ContextDoc] = []
    for row in rows:
        r = row if isinstance(row, dict) else dict(zip(columns, row))
        label = _SECTION_LABEL.get(r["section"], r["section"])
        fy = f" FY{r['fiscal_year']}" if r.get("fiscal_year") else ""
        docs.append(
            ContextDoc(
                doc_type=f"filing:{r['section']}",
                title=f"{r['ticker']} 10-K {label}{fy}",
                content=r["content"],
                similarity=float(r["similarity"]),
            )
        )

    logger.info(
        "filing_chunks_retrieved",
        question=question,
        company_id=company_id,
        k=k,
        titles=[d.title for d in docs],
        top_similarity=docs[0].similarity if docs else None,
    )
    return docs


def resolve_company(question: str) -> tuple[int | None, str | None]:
    """Best-effort map a question to a single company_id + ticker.

    Matches an uppercase ticker token first (exact), then a company-name
    substring. Returns (None, None) when the question names no company or
    names several, in which case retrieval searches across all filings.
    """
    import re

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, ticker, name FROM companies")
            raw = cur.fetchall()

    rows = [
        r if isinstance(r, dict) else {"id": r[0], "ticker": r[1], "name": r[2]}
        for r in raw
    ]
    q_upper = question.upper()
    q_lower = question.lower()

    # 1) exact ticker as a whole word
    ticker_hits = [
        c for c in rows if re.search(rf"\b{re.escape(c['ticker'])}\b", q_upper)
    ]
    if len(ticker_hits) == 1:
        return ticker_hits[0]["id"], ticker_hits[0]["ticker"]

    # 2) company-name match: compare the distinctive first word of the name
    #    (e.g. "apple" from "Apple Inc.", "jpmorgan" from "JPMORGAN CHASE & CO")
    name_hits = []
    for c in rows:
        head = re.sub(r"[^a-z]", "", c["name"].split()[0].lower())
        if len(head) >= 4 and head in q_lower:
            name_hits.append(c)
    if len(name_hits) == 1:
        return name_hits[0]["id"], name_hits[0]["ticker"]

    return None, None
