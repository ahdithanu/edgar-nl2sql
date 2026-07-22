"""The RAG context corpus: everything Claude needs to write correct SQL.

Why this file matters more than it looks
----------------------------------------
This system does retrieval-BEFORE-generation: when a question arrives, we
embed it, pull the top-k most similar documents from this corpus (stored in
`rag_documents` with pgvector), and inject them verbatim into the SQL
generation prompt. The model never sees the live database — these documents
ARE its knowledge of the schema, the data's quirks, and the SQL idioms that
work. Every eval failure traces back to either a missing document here or a
document that didn't say the crucial thing plainly.

Design principles for the docs below:
- One concept per document, so cosine similarity can match narrowly phrased
  questions ("net margin", "biggest company", "Q3 revenue") to exactly the
  right guidance.
- State exact values (metric names, fiscal_period values, ticker symbols)
  rather than describing them — the model copies literals out of context.
- Include worked SQL for every non-trivial pattern (ratios, YoY,
  superlatives). Models imitate examples far more reliably than prose.
- Repeat the two data gotchas (flow vs instant semantics; derived Q4)
  anywhere they could bite, because only ~8 docs are retrieved per question.

Consumed by scripts/build_embeddings.py, which embeds each doc with Voyage
(input_type="document") and upserts into rag_documents keyed on title.
"""

from textwrap import dedent


def _doc(doc_type: str, title: str, content: str) -> dict:
    """Normalize indentation so docs read cleanly both here and in prompts."""
    return {"doc_type": doc_type, "title": title, "content": dedent(content).strip()}


CONTEXT_DOCS: list[dict] = [
    # ------------------------------------------------------------------
    # table_schema (3): one per table — DDL + semantics + join keys
    # ------------------------------------------------------------------
    _doc(
        "table_schema",
        "companies table schema",
        """
        Table `companies` — one row per SEC-registered company (25 rows total).

        CREATE TABLE companies (
            id               SERIAL PRIMARY KEY,
            cik              TEXT NOT NULL UNIQUE,  -- zero-padded 10-digit SEC CIK, e.g. '0000320193'
            ticker           TEXT NOT NULL UNIQUE,  -- uppercase stock symbol, e.g. 'AAPL'
            name             TEXT NOT NULL,         -- SEC registrant name, e.g. 'Apple Inc.'
            sic_description  TEXT                   -- SEC industry classification, e.g. 'Electronic Computers'
        );

        Join key: companies.id = financial_metrics.company_id and
        companies.id = filings.company_id. Almost every query starts with:
            FROM financial_metrics fm JOIN companies c ON c.id = fm.company_id
        Filter companies by `ticker` (exact uppercase match) when the question
        names a ticker, or by `name ILIKE '%...%'` for a company name.
        """,
    ),
    _doc(
        "table_schema",
        "filings table schema",
        """
        Table `filings` — one row per SEC filing (10-K annual report or 10-Q
        quarterly report) that supplied metric values.

        CREATE TABLE filings (
            id                SERIAL PRIMARY KEY,
            company_id        INT NOT NULL REFERENCES companies(id),
            accession_number  TEXT NOT NULL UNIQUE,  -- SEC accession id, e.g. '0000320193-23-000106'
            form              TEXT NOT NULL,         -- '10-K' (annual) or '10-Q' (quarterly)
            filed_date        DATE NOT NULL,         -- when the filing hit EDGAR
            fiscal_year       INT NOT NULL,
            fiscal_period     TEXT NOT NULL          -- 'FY','Q1','Q2','Q3','Q4'
        );

        Join keys: filings.company_id = companies.id;
        financial_metrics.filing_id = filings.id.
        Most metric questions do NOT need this table — join it only when the
        question asks about the filing itself (form type, filing date,
        accession number), e.g. "When did Apple file its fiscal 2023 10-K?"
        """,
    ),
    _doc(
        "table_schema",
        "financial_metrics table schema",
        """
        Table `financial_metrics` — the fact table. One row per company per
        metric per fiscal period. This is where all numeric answers live.

        CREATE TABLE financial_metrics (
            id             SERIAL PRIMARY KEY,
            company_id     INT NOT NULL REFERENCES companies(id),
            filing_id      INT REFERENCES filings(id),
            metric         TEXT NOT NULL,   -- exactly one of: 'revenue','net_income','total_assets','total_liabilities','eps_diluted'
            fiscal_year    INT NOT NULL,    -- 2020..2024 (company fiscal year label)
            fiscal_period  TEXT NOT NULL,   -- 'FY','Q1','Q2','Q3','Q4'
            value          NUMERIC NOT NULL,-- raw units: whole US dollars, or dollars-per-share for eps_diluted
            unit           TEXT NOT NULL,   -- 'USD' or 'USD/share'
            start_date     DATE,            -- period start (NULL for balance-sheet snapshots)
            end_date       DATE             -- period end / snapshot date
        );
        -- UNIQUE (company_id, metric, fiscal_year, fiscal_period)

        Canonical lookup pattern:
            SELECT fm.value
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'AAPL'
              AND fm.metric = 'revenue'
              AND fm.fiscal_year = 2023
              AND fm.fiscal_period = 'FY';

        ALWAYS filter all three of metric, fiscal_year, and fiscal_period —
        omitting fiscal_period mixes annual and quarterly rows and inflates
        sums by roughly 2x.
        """,
    ),
    # ------------------------------------------------------------------
    # column (~8): the tricky columns
    # ------------------------------------------------------------------
    _doc(
        "column",
        "financial_metrics.metric — the five exact metric values",
        """
        Column financial_metrics.metric holds EXACTLY these five strings
        (lowercase, snake_case — any other spelling matches zero rows):

        - 'revenue'            total revenue / net sales / top line (unit 'USD')
        - 'net_income'         net income / profit / earnings / bottom line (unit 'USD')
        - 'total_assets'       balance-sheet total assets (unit 'USD')
        - 'total_liabilities'  balance-sheet total liabilities (unit 'USD')
        - 'eps_diluted'        diluted earnings per share (unit 'USD/share')

        Mapping natural language to metric values:
        "sales", "turnover", "top line" -> 'revenue'
        "profit", "earnings", "income", "bottom line" -> 'net_income'
        "assets" -> 'total_assets'; "liabilities", "debt" (loosely) -> 'total_liabilities'
        "EPS", "earnings per share" -> 'eps_diluted'
        There are NO other metrics (no gross margin, cash, dividends, or share
        count columns) — derived quantities must be computed from these five.
        """,
    ),
    _doc(
        "column",
        "financial_metrics.fiscal_period — FY vs quarters, flow vs instant, derived Q4",
        """
        Column financial_metrics.fiscal_period is one of 'FY','Q1','Q2','Q3','Q4'.
        Its meaning depends on the metric type:

        FLOW metrics ('revenue', 'net_income', 'eps_diluted') accumulate over
        a period. Rows exist for FY and Q1..Q4 per fiscal year:
        - 'FY' is the full-year total reported in the 10-K. For annual
          questions use fiscal_period = 'FY' — do NOT sum the four quarters.
        - 'Q4' is DERIVED as FY - (Q1 + Q2 + Q3) because companies file a
          10-K instead of a fourth 10-Q. Exact for dollar amounts; for
          eps_diluted it is an approximation (share counts vary by quarter).

        INSTANT (balance-sheet) metrics ('total_assets', 'total_liabilities')
        are point-in-time snapshots. Rows exist ONLY for Q1, Q2, Q3, and FY:
        - The 'FY' row IS the fiscal-year-end balance; there is no 'Q4' row.
          A query for total_assets with fiscal_period = 'Q4' returns zero rows.
        - Never SUM balance snapshots across periods — a year-end balance is
          the FY row, not an aggregate.

        Rule of thumb: "for the year" => fiscal_period = 'FY';
        "in Q3" => fiscal_period = 'Q3'; year-end balances => 'FY'.
        """,
    ),
    _doc(
        "column",
        "financial_metrics.fiscal_year — company fiscal year labels, not calendar years",
        """
        Column financial_metrics.fiscal_year (INT, 2020..2024) is the
        COMPANY'S OWN fiscal year label, which may not match the calendar year:
        - AAPL fiscal 2023 ended late September 2023.
        - MSFT fiscal 2024 ended June 2024.
        - NVDA labels years ahead: NVDA fiscal 2024 ended January 2024
          (mostly calendar 2023).
        - WMT and HD fiscal years end in late January / early February.
        - COST ends around end of August; DIS around end of September.
        Most other covered companies (AMZN, GOOGL, META, TSLA, JPM, banks,
        energy, pharma) use calendar years ending December 31.

        When a question says "in 2023", interpret it as fiscal_year = 2023
        unless it explicitly says "calendar year". To be precise about actual
        dates, check start_date/end_date. Available range: fiscal 2020-2024;
        the latest complete year is fiscal 2024.
        """,
    ),
    _doc(
        "column",
        "financial_metrics.value and unit — raw dollars, not millions",
        """
        Column financial_metrics.value is NUMERIC in RAW units:
        - unit = 'USD': whole US dollars. Apple's fiscal 2023 revenue is
          stored as 383285000000 (383.285 billion), NOT 383285 or 383.285.
        - unit = 'USD/share' (only for metric 'eps_diluted'): dollars per
          share, e.g. 6.13.

        Implications for SQL:
        - Never multiply or divide by 1e6/1e9 inside WHERE comparisons unless
          the question gives a threshold ("over $100 billion" => value > 100e9).
        - For readability you MAY project value / 1e9 AS value_billions, but
          keep filters in raw dollars.
        - Never compare or add 'USD' values with 'USD/share' values; they are
          different dimensions.
        """,
    ),
    _doc(
        "column",
        "companies.ticker vs companies.name — how to match a company",
        """
        Two ways to identify a company; prefer ticker when you know it:
        - companies.ticker: uppercase symbol, exact match: WHERE c.ticker = 'AAPL'.
        - companies.name: SEC registrant name; match loosely with ILIKE
          because legal names differ from common names:
          WHERE c.name ILIKE '%apple%'.

        Careful mappings (common name -> ticker): Apple->AAPL,
        Microsoft->MSFT, Google/Alphabet->GOOGL, Amazon->AMZN, NVIDIA->NVDA,
        Facebook/Meta->META, Tesla->TSLA, JPMorgan/Chase->JPM,
        Bank of America->BAC, Goldman Sachs->GS, Visa->V, Mastercard->MA,
        Walmart->WMT, Costco->COST, Home Depot->HD, Coca-Cola/Coke->KO,
        Pepsi/PepsiCo->PEP, McDonald's->MCD, Exxon/ExxonMobil->XOM,
        Chevron->CVX, Johnson & Johnson/J&J->JNJ, Pfizer->PFE,
        UnitedHealth->UNH, Disney->DIS, Netflix->NFLX.
        Note the one-letter ticker 'V' (Visa) and two-letter 'MA' (Mastercard):
        always quote tickers as string literals.
        """,
    ),
    _doc(
        "column",
        "financial_metrics.start_date and end_date — period windows",
        """
        Columns start_date / end_date (DATE) give the actual calendar window
        each row covers:
        - Flow rows (revenue, net_income, eps_diluted): start_date..end_date
          spans the period — roughly 90 days for quarters, roughly 365 days
          for FY rows.
        - Instant rows (total_assets, total_liabilities): start_date is NULL;
          end_date is the balance-sheet snapshot date.

        Use these when a question needs real calendar dates ("revenue for the
        quarter ending in June 2023"): filter on end_date ranges, e.g.
        WHERE fm.end_date BETWEEN '2023-06-01' AND '2023-07-31'.
        For ordinary fiscal-period questions, prefer fiscal_year +
        fiscal_period filters — they are indexed and unambiguous.
        """,
    ),
    _doc(
        "column",
        "companies.sic_description — industry classification",
        """
        Column companies.sic_description holds the SEC's SIC industry label,
        e.g. 'Electronic Computers' (AAPL), 'Services-Prepackaged Software'
        (MSFT), 'National Commercial Banks' (JPM, BAC), 'Petroleum Refining'
        (XOM, CVX), 'Pharmaceutical Preparations' (JNJ, PFE).

        These labels are narrow and idiosyncratic — for "tech companies" or
        "banks" style questions, match broadly with ILIKE:
            WHERE c.sic_description ILIKE '%bank%'
        or simply enumerate the tickers you mean:
            WHERE c.ticker IN ('JPM', 'BAC', 'GS')
        Enumerating tickers is usually more reliable than trusting SIC text.
        """,
    ),
    _doc(
        "glossary",
        "sector and industry questions — banks, tech, energy, healthcare",
        """
        Questions about a GROUP of companies rather than named ones: "all
        banks", "the banks", "tech companies", "energy companies", "pharma",
        "healthcare", "retailers", "which industry", "compare sectors",
        "average revenue growth across banks", "how profitable is the tech
        sector", "biggest bank by assets".

        These ARE answerable. companies.sic_description holds an SEC industry
        label for every company in the database — there is no separate sector
        table, but the column is populated for all 25 companies, so never
        answer that industry information is unavailable.

        WHY THIS DOC EXISTS: questions like "average revenue growth across all
        banks" mix a sector filter with a metric concept, and retrieval tends
        to return only growth/metric docs — leaving the model believing no
        industry data exists. It does. Match broadly with ILIKE:

            -- every bank's revenue growth 2022 -> 2023
            SELECT c.ticker, c.name,
                   ROUND(100.0 * (r23.value - r22.value) / r22.value, 1) AS growth_pct
            FROM companies c
            JOIN financial_metrics r22 ON r22.company_id = c.id
             AND r22.metric = 'revenue' AND r22.fiscal_year = 2022 AND r22.fiscal_period = 'FY'
            JOIN financial_metrics r23 ON r23.company_id = c.id
             AND r23.metric = 'revenue' AND r23.fiscal_year = 2023 AND r23.fiscal_period = 'FY'
            WHERE c.sic_description ILIKE '%bank%'
            ORDER BY growth_pct DESC;

        Useful ILIKE patterns: '%bank%' (JPM, BAC), '%software%' or
        '%computer%' (MSFT, AAPL), '%petroleum%' (XOM, CVX),
        '%pharmaceutical%' (JNJ, PFE), '%retail%' or '%stores%' (WMT, COST, HD).
        For finance broadly, include brokers: sic_description ILIKE '%bank%'
        OR sic_description ILIKE '%security brokers%' (GS).

        If a sector label genuinely matches no company, say so — but check with
        ILIKE first rather than assuming the classification does not exist.
        """,
    ),
    _doc(
        "column",
        "financial_metrics.filing_id — linking metrics to their source filing",
        """
        Column financial_metrics.filing_id references filings.id: the SEC
        filing the value came from. Annual (FY) rows and derived Q4 rows point
        at the 10-K; Q1-Q3 rows point at 10-Qs.

        Only join filings when provenance is asked about:
            SELECT fm.value, f.form, f.filed_date, f.accession_number
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            JOIN filings f  ON f.id = fm.filing_id
            WHERE c.ticker = 'MSFT' AND fm.metric = 'net_income'
              AND fm.fiscal_year = 2024 AND fm.fiscal_period = 'FY';
        filing_id can be NULL in edge cases — use LEFT JOIN if rows must not drop.
        """,
    ),
    # ------------------------------------------------------------------
    # glossary (~18): financial concepts + worked SQL patterns
    # ------------------------------------------------------------------
    _doc(
        "glossary",
        "revenue — definition (top line)",
        """
        Revenue (also: sales, net sales, turnover, the "top line") is the
        total money a company earned from its business before any costs.
        Stored as metric = 'revenue', unit 'USD', raw dollars. Flow metric:
        FY and Q1..Q4 rows exist (Q4 derived as FY minus first three quarters).

        "What was Amazon's revenue in fiscal 2022?":
            SELECT fm.value
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'AMZN' AND fm.metric = 'revenue'
              AND fm.fiscal_year = 2022 AND fm.fiscal_period = 'FY';
        """,
    ),
    _doc(
        "glossary",
        "net income — definition (bottom line)",
        """
        Net income (also: profit, net profit, earnings, the "bottom line") is
        what remains of revenue after all expenses and taxes. It can be
        negative (a net loss). Stored as metric = 'net_income', unit 'USD'.
        Flow metric: FY and Q1..Q4 rows (Q4 derived).

        "How much profit did Tesla make in fiscal 2021?":
            SELECT fm.value
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'TSLA' AND fm.metric = 'net_income'
              AND fm.fiscal_year = 2021 AND fm.fiscal_period = 'FY';
        "Which companies lost money (net loss) in fiscal 2020?" adds
        AND fm.value < 0.
        """,
    ),
    _doc(
        "glossary",
        "total assets and total liabilities — balance sheet snapshots",
        """
        Total assets = everything a company owns; total liabilities =
        everything it owes. Both are INSTANT (point-in-time) balance-sheet
        metrics: metric = 'total_assets' / 'total_liabilities', unit 'USD'.

        Rows exist for fiscal_period 'Q1','Q2','Q3','FY' only. The FY row is
        the fiscal-year-end snapshot — there is NO 'Q4' row for these metrics,
        and summing snapshots across periods is meaningless.

        "What were JPMorgan's total assets at the end of fiscal 2023?":
            SELECT fm.value
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'JPM' AND fm.metric = 'total_assets'
              AND fm.fiscal_year = 2023 AND fm.fiscal_period = 'FY';
        """,
    ),
    _doc(
        "glossary",
        "shareholders equity — derived as assets minus liabilities",
        """
        Shareholders' equity (book value, net worth) is NOT stored directly.
        Approximate it as total_assets - total_liabilities for the same
        company and period. Both metrics live in separate rows, so combine
        them with conditional aggregation:

        "What was Microsoft's shareholders' equity at fiscal 2023 year end?":
            SELECT
              SUM(fm.value) FILTER (WHERE fm.metric = 'total_assets')
            - SUM(fm.value) FILTER (WHERE fm.metric = 'total_liabilities')
              AS shareholders_equity
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'MSFT'
              AND fm.metric IN ('total_assets', 'total_liabilities')
              AND fm.fiscal_year = 2023 AND fm.fiscal_period = 'FY';

        Caveat worth stating in answers: this is an accounting identity
        (assets = liabilities + equity), so it is a close approximation, not
        an independently reported figure.
        """,
    ),
    _doc(
        "glossary",
        "EPS — diluted earnings per share",
        """
        EPS (earnings per share) here is DILUTED EPS: net income divided by
        the weighted-average diluted share count, as reported by the company.
        Stored as metric = 'eps_diluted', unit 'USD/share' (e.g. 6.13 means
        $6.13 per share). Flow metric with FY and Q1..Q4 rows.

        Caveat: the Q4 EPS row is derived as FY - (Q1+Q2+Q3), which is only
        approximately right because each quarter uses its own share count —
        mention this if a question hinges on Q4 EPS precision.

        "What was Apple's diluted EPS in fiscal 2023?":
            SELECT fm.value
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'AAPL' AND fm.metric = 'eps_diluted'
              AND fm.fiscal_year = 2023 AND fm.fiscal_period = 'FY';
        Do not recompute EPS from net_income (no share-count column exists).
        """,
    ),
    _doc(
        "glossary",
        "net margin / profit margin — SQL pattern (two rows, one ratio)",
        """
        Net margin (profit margin) = net_income / revenue for the SAME company
        and SAME period. The two numbers live in two different rows of
        financial_metrics, so a ratio needs conditional aggregation (FILTER)
        or a self-join — FILTER is cleaner:

        "What was Apple's net margin in fiscal 2023?":
            SELECT
              SUM(fm.value) FILTER (WHERE fm.metric = 'net_income')
              / NULLIF(SUM(fm.value) FILTER (WHERE fm.metric = 'revenue'), 0)
              AS net_margin
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'AAPL'
              AND fm.metric IN ('net_income', 'revenue')
              AND fm.fiscal_year = 2023 AND fm.fiscal_period = 'FY';

        Result is a fraction (0.253 = 25.3%); multiply by 100 for percent.
        For "which company had the highest net margin", GROUP BY c.ticker,
        c.name and ORDER BY the ratio DESC LIMIT 1. Always NULLIF the
        denominator to avoid division by zero.
        """,
    ),
    _doc(
        "glossary",
        "year-over-year (YoY) growth — SQL pattern",
        """
        YoY growth compares a metric to the same period one fiscal year
        earlier: (current - prior) / prior. Self-join the metrics table on
        fiscal_year - 1 with the same company, metric, and fiscal_period:

        "How much did NVIDIA's revenue grow in fiscal 2024?":
            SELECT
              cur.value AS current_revenue,
              prev.value AS prior_revenue,
              (cur.value - prev.value) / NULLIF(prev.value, 0) * 100 AS yoy_growth_pct
            FROM financial_metrics cur
            JOIN financial_metrics prev
              ON prev.company_id = cur.company_id
             AND prev.metric = cur.metric
             AND prev.fiscal_period = cur.fiscal_period
             AND prev.fiscal_year = cur.fiscal_year - 1
            JOIN companies c ON c.id = cur.company_id
            WHERE c.ticker = 'NVDA' AND cur.metric = 'revenue'
              AND cur.fiscal_year = 2024 AND cur.fiscal_period = 'FY';

        For "fastest growing company", compute this per company (GROUP BY is
        unnecessary — the self-join already yields one row per company) and
        ORDER BY yoy_growth_pct DESC LIMIT 1. Growth from a negative or zero
        base is not meaningful — NULLIF guards the division.
        """,
    ),
    _doc(
        "glossary",
        "debt ratio / leverage — SQL pattern",
        """
        Debt ratio (leverage) = total_liabilities / total_assets, same company
        and period. Values near 1.0 mean the company is highly leveraged
        (banks routinely ~0.9); low values mean asset-rich balance sheets.
        Both are instant metrics — use fiscal_period = 'FY' for year-end.

        "Which company was most leveraged at the end of fiscal 2023?":
            SELECT c.ticker, c.name,
              SUM(fm.value) FILTER (WHERE fm.metric = 'total_liabilities')
              / NULLIF(SUM(fm.value) FILTER (WHERE fm.metric = 'total_assets'), 0)
              AS debt_ratio
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE fm.metric IN ('total_liabilities', 'total_assets')
              AND fm.fiscal_year = 2023 AND fm.fiscal_period = 'FY'
            GROUP BY c.ticker, c.name
            ORDER BY debt_ratio DESC
            LIMIT 1;
        """,
    ),
    _doc(
        "glossary",
        "return on assets (ROA) — SQL pattern mixing a flow and an instant",
        """
        ROA = net_income (a flow over the fiscal year, fiscal_period 'FY')
        divided by total_assets (a snapshot; use the same year's 'FY' row as
        the year-end approximation). Both rows share fiscal_year and
        fiscal_period = 'FY', so FILTER aggregation still works:

        "What was Walmart's return on assets in fiscal 2024?":
            SELECT
              SUM(fm.value) FILTER (WHERE fm.metric = 'net_income')
              / NULLIF(SUM(fm.value) FILTER (WHERE fm.metric = 'total_assets'), 0)
              AS roa
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'WMT'
              AND fm.metric IN ('net_income', 'total_assets')
              AND fm.fiscal_year = 2024 AND fm.fiscal_period = 'FY';

        (Purists average beginning and ending assets; with this schema the
        year-end snapshot is the standard simplification — say so in answers.)
        """,
    ),
    _doc(
        "glossary",
        "superlatives — biggest, largest, highest, most profitable",
        """
        "Biggest/largest company" by default means highest REVENUE; "most
        profitable" means highest NET INCOME (or highest net margin if the
        question says margin/percentage). "Most valuable" (market cap) is NOT
        answerable from this data — no stock prices exist.

        Pattern: filter to ONE metric, ONE fiscal_year, fiscal_period = 'FY',
        then ORDER BY value DESC LIMIT 1 (or LIMIT N for "top N"):

        "Which company had the highest revenue in fiscal 2024?":
            SELECT c.name, c.ticker, fm.value AS revenue
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE fm.metric = 'revenue'
              AND fm.fiscal_year = 2024 AND fm.fiscal_period = 'FY'
            ORDER BY fm.value DESC
            LIMIT 1;

        For "smallest"/"lowest", ORDER BY fm.value ASC. Forgetting the
        fiscal_period filter is the classic bug — it double-counts by mixing
        FY rows with quarterly rows.
        """,
    ),
    _doc(
        "glossary",
        "multi-company comparison — SQL pattern",
        """
        "Compare X and Y" questions: filter tickers with IN, return one
        labeled row per company rather than doing arithmetic in your head:

        "Compare Coca-Cola and PepsiCo revenue in fiscal 2023":
            SELECT c.ticker, c.name, fm.value AS revenue
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker IN ('KO', 'PEP')
              AND fm.metric = 'revenue'
              AND fm.fiscal_year = 2023 AND fm.fiscal_period = 'FY'
            ORDER BY fm.value DESC;

        For side-by-side multi-metric comparisons, pivot with FILTER:
            SELECT c.ticker,
              SUM(fm.value) FILTER (WHERE fm.metric = 'revenue')    AS revenue,
              SUM(fm.value) FILTER (WHERE fm.metric = 'net_income') AS net_income
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker IN ('XOM', 'CVX')
              AND fm.metric IN ('revenue', 'net_income')
              AND fm.fiscal_year = 2023 AND fm.fiscal_period = 'FY'
            GROUP BY c.ticker;
        """,
    ),
    _doc(
        "glossary",
        "quarterly lookups — SQL pattern",
        """
        Quarterly questions use fiscal_period 'Q1','Q2','Q3','Q4'. Remember
        these are the COMPANY's fiscal quarters — Apple's Q1 ends in late
        December, Microsoft's Q1 ends in September.

        "What was Apple's revenue in Q1 of fiscal 2024?":
            SELECT fm.value
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'AAPL' AND fm.metric = 'revenue'
              AND fm.fiscal_year = 2024 AND fm.fiscal_period = 'Q1';

        "Show Microsoft's quarterly net income across fiscal 2023":
            SELECT fm.fiscal_period, fm.value
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'MSFT' AND fm.metric = 'net_income'
              AND fm.fiscal_year = 2023
              AND fm.fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
            ORDER BY fm.fiscal_period;

        Caveats: flow Q4 rows are derived (FY minus first three quarters);
        balance metrics (total_assets/total_liabilities) have NO Q4 row —
        use 'FY' for their year-end value.
        """,
    ),
    _doc(
        "glossary",
        "quarter-over-quarter (QoQ) growth — SQL pattern",
        """
        QoQ growth compares consecutive quarters. Because quarters are labels
        ('Q1'..'Q4'), order rows by their actual end_date and use LAG:

        "How did Netflix's revenue change quarter over quarter in fiscal 2023?":
            SELECT fiscal_year, fiscal_period, value,
                   (value - LAG(value) OVER (ORDER BY end_date))
                   / NULLIF(LAG(value) OVER (ORDER BY end_date), 0) * 100
                   AS qoq_growth_pct
            FROM financial_metrics fm
            JOIN companies c ON c.id = fm.company_id
            WHERE c.ticker = 'NFLX' AND fm.metric = 'revenue'
              AND fm.fiscal_year = 2023
              AND fm.fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
            ORDER BY end_date;

        Ordering by end_date (not by the period string) is what makes the
        window correct — it also works across fiscal-year boundaries if you
        widen the fiscal_year filter.
        """,
    ),
    _doc(
        "glossary",
        "latest year available — data covers fiscal 2020 through fiscal 2024",
        """
        The database contains fiscal years 2020 through 2024 only. When a
        question says "latest", "most recent", "current", or "last year",
        use fiscal_year = 2024 (the latest complete fiscal year for every
        covered company). There is no fiscal 2025 or partial-year data, and
        nothing before fiscal 2020.

        A robust alternative that never hardcodes the year:
            WHERE fm.fiscal_year = (SELECT MAX(fiscal_year) FROM financial_metrics)

        If a question asks about a year outside 2020-2024, the correct
        behavior is to say the data does not cover it, not to silently
        substitute a different year.
        """,
    ),
    _doc(
        "glossary",
        "trailing twelve months (TTM) — caveat and closest approximation",
        """
        TTM ("trailing twelve months", "last 12 months") normally means the
        four most recent quarters regardless of fiscal-year boundaries. This
        database is organized by fiscal year, so true rolling TTM as of an
        arbitrary date is awkward; the practical answer is the latest full
        fiscal year (fiscal_period = 'FY', fiscal_year = 2024) — state that
        substitution explicitly in the answer.

        If genuinely needed, sum the four quarterly flow rows with the latest
        end_date:
            SELECT SUM(value) FROM (
              SELECT value
              FROM financial_metrics fm
              JOIN companies c ON c.id = fm.company_id
              WHERE c.ticker = 'AAPL' AND fm.metric = 'revenue'
                AND fm.fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4')
              ORDER BY fm.end_date DESC
              LIMIT 4
            ) last_four;
        Only valid for flow metrics — never sum balance-sheet snapshots.
        """,
    ),
    _doc(
        "glossary",
        "FY is not the sum of quarters — prefer fiscal_period = 'FY'",
        """
        For annual totals ALWAYS use the fiscal_period = 'FY' row rather than
        SUM over Q1..Q4, even though both should be close:
        - The FY row is the audited figure straight from the 10-K.
        - Q4 is derived (FY minus first three quarters), so summing quarters
          just reconstructs FY with extra steps and rounding noise — and for
          eps_diluted the quarterly sum can differ noticeably from the true
          FY figure.
        - For total_assets/total_liabilities, summing periods is meaningless
          (snapshots, not flows) — the FY row alone is the year-end balance.

        Wrong:  SELECT SUM(value) ... WHERE fiscal_period IN ('Q1','Q2','Q3','Q4')
        Right:  SELECT value       ... WHERE fiscal_period = 'FY'
        """,
    ),
    _doc(
        "glossary",
        "companies covered — the 25 tickers in this database",
        """
        Exactly 25 large-cap US companies are covered. Questions about any
        other company (e.g. Intel, Berkshire, Oracle) cannot be answered —
        say so rather than guessing a substitute.

        Tech: AAPL (Apple), MSFT (Microsoft), GOOGL (Alphabet/Google),
        AMZN (Amazon), NVDA (NVIDIA), META (Meta/Facebook), TSLA (Tesla),
        NFLX (Netflix).
        Financials: JPM (JPMorgan Chase), BAC (Bank of America),
        GS (Goldman Sachs), V (Visa), MA (Mastercard).
        Consumer/retail: WMT (Walmart), COST (Costco), HD (Home Depot),
        KO (Coca-Cola), PEP (PepsiCo), MCD (McDonald's), DIS (Disney).
        Energy: XOM (Exxon Mobil), CVX (Chevron).
        Healthcare: JNJ (Johnson & Johnson), PFE (Pfizer),
        UNH (UnitedHealth Group).

        "FAANG", "big tech", "banks" style groupings should be translated to
        explicit ticker lists with c.ticker IN (...).
        """,
    ),
    _doc(
        "glossary",
        "common query pitfalls — checklist before finalizing SQL",
        """
        The five bugs that account for most wrong answers on this schema:

        1. Missing fiscal_period filter -> FY and quarterly rows mixed
           together, roughly doubling sums. Always pin fiscal_period.
        2. Wrong metric spelling -> zero rows. Only five values exist:
           'revenue','net_income','total_assets','total_liabilities',
           'eps_diluted' (lowercase snake_case).
        3. Asking balance metrics for 'Q4' -> zero rows. Year-end balances
           live in the 'FY' row for total_assets/total_liabilities.
        4. Ratio computed across mismatched periods -> nonsense. Both sides
           of net margin / debt ratio / ROA must share company_id,
           fiscal_year, AND fiscal_period.
        5. Unquoted or lowercased tickers -> zero rows ('V' and 'MA' are
           easy to mangle). Tickers are uppercase string literals.

        If a query legitimately returns zero rows, the likely causes are, in
        order: misspelled metric, impossible fiscal_period for the metric,
        a company outside the 25 covered, or a year outside 2020-2024.
        """,
    ),
]

# Fail fast at import time if a title collides — rag_documents keys on title,
# and a silent overwrite in build_embeddings would drop a doc from the corpus.
_titles = [d["title"] for d in CONTEXT_DOCS]
assert len(_titles) == len(set(_titles)), "CONTEXT_DOCS titles must be unique"
