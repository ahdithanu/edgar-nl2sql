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


def _describe_chunk(
    form: str, ticker: str, section: str, fiscal_year: int | None
) -> tuple[str, str]:
    """Return (doc_type, citation title) for a narrative chunk.

    One retrieval path serves two document kinds, so the citation is derived
    from `form`: a 10-K section keeps its exact historical title
    ("AAPL 10-K Risk Factors FY2024"), while an earnings-call chunk reads
    "AAPL Earnings Call Q1 FY2024". doc_type lets a reader tell the sources
    apart in the response (filing:* vs transcript:*).
    """
    fy = f" FY{fiscal_year}" if fiscal_year else ""
    if form == "earnings_call":
        quarter = section.rsplit("_", 1)[-1].upper()  # 'earnings_q1' -> 'Q1'
        return f"transcript:{section}", f"{ticker} Earnings Call {quarter}{fy}"
    label = _SECTION_LABEL.get(section, section)
    return f"filing:{section}", f"{ticker} 10-K {label}{fy}"


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
               fc.form,
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
        doc_type, title = _describe_chunk(
            r.get("form") or "10-K", r["ticker"], r["section"], r.get("fiscal_year")
        )
        docs.append(
            ContextDoc(
                doc_type=doc_type,
                title=title,
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


def _match_company(
    question: str, companies: list[dict]
) -> tuple[int | None, str | None]:
    """Pure matcher: map a question to a single (company_id, ticker) or (None, None).

    Order:
      1. A ticker written as an uppercase whole word in the ORIGINAL question
         ("AAPL", "NVDA"). Case-SENSITIVE on purpose: tickers are conventionally
         uppercase, and a case-insensitive match makes short tickers collide with
         ordinary words — "ON" (ON Semiconductor) would match the word "on",
         "IT", "ALL", "KEY" likewise. Requiring the uppercase form kills that
         whole class of false positives while still catching how people actually
         write tickers.
      2. Otherwise, the distinctive first word of a company name ("apple" from
         "Apple Inc.", "nvidia" from "NVIDIA CORP"), matched case-insensitively.

    Returns (None, None) when zero or several companies match, so retrieval then
    searches across all filings rather than guessing one.
    """
    import re

    ticker_hits = [
        c for c in companies
        if re.search(rf"\b{re.escape(c['ticker'])}\b", question)  # case-sensitive
    ]
    if len(ticker_hits) == 1:
        return ticker_hits[0]["id"], ticker_hits[0]["ticker"]

    q_lower = question.lower()
    name_hits = []
    for c in companies:
        head = re.sub(r"[^a-z]", "", c["name"].split()[0].lower())
        if len(head) >= 4 and head in q_lower:
            name_hits.append(c)
    if len(name_hits) == 1:
        return name_hits[0]["id"], name_hits[0]["ticker"]

    return None, None


def resolve_company(question: str) -> tuple[int | None, str | None]:
    """Best-effort map a question to a single company_id + ticker.

    Fetches the company list and delegates to _match_company. Returns
    (None, None) when the question names no company or names several, in which
    case retrieval searches across all filings.
    """
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, ticker, name FROM companies")
            raw = cur.fetchall()

    companies = [
        r if isinstance(r, dict) else {"id": r[0], "ticker": r[1], "name": r[2]}
        for r in raw
    ]
    return _match_company(question, companies)
