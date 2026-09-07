# RAG corpus audit — 2026-09-07

All 31 documents in `rag_documents` audited for stale scope, count and
coverage claims. Findings recorded **before** any correction was made.

## How this surfaced

Found by [driftgate](../../Documents/ChatGPT/Driftgate), an eval harness that
runs this app's exact prompt contract against a frozen snapshot. Two suite
cases — "List the fire, marine and casualty insurance companies" and "What were
the combined total assets of REITs in fiscal 2024?" — were refused by three
independently-trained models (`claude-sonnet-4-6`, `claude-sonnet-5`,
`claude-haiku-4-5`). Every refusal restated the same premise: that the database
covers only 25 companies. Retrieval was working correctly; the retrieved
document was wrong.

## Verification against the live database

Queried directly, not inferred:

| Claim under test | Measured |
|---|---|
| `companies` row count | **478** |
| distinct `sic_description` values | **153** |
| Berkshire Hathaway present? | **yes** — `BRK-B`, "BERKSHIRE HATHAWAY INC", *Fire, Marine & Casualty Insurance* |
| Intel present? | **yes** — `INTC`, "INTEL CORP" |
| Oracle present? | **yes** — `ORCL`, "ORACLE CORP" |
| companies matching `'%fire, marine%'` | **9** |
| fiscal years loaded | 2020–2026 |
| companies with metrics | 364 |
| `rag_documents` | 31 |

Intel, Berkshire and Oracle are the three companies doc 243 names as examples
of what "cannot be answered". All three are in the data.

## Defective documents (3 of 31)

### doc 215 — `companies table schema` (table_schema)
> Table `companies`, one row per SEC-registered company (**25 rows total**).

478 rows. Stale count in an otherwise accurate schema document.

### doc 225 — `sector and industry questions: banks, tech, energy, healthcare` (glossary)
> …there is no separate sector table, but the column is populated for **all 25
> companies**, so never answer that industry information is unavailable.

478 companies. The document's *intent* — that industry data exists and sector
questions are answerable — is correct; only the count is stale.

### doc 243 — `companies covered: the 25 tickers in this database` (glossary)
> **Exactly 25** large-cap US companies are covered. Questions about any other
> company (e.g. **Intel, Berkshire, Oracle**) cannot be answered, say so rather
> than guessing a substitute.

Three defects in one document:
1. The count (25) is wrong — 478.
2. The three named counter-examples are all present in the data.
3. The enumerated ticker list is presented as exhaustive; it lists 25 of 478.

This is the document that caused the observed refusals. It is retrieved at
rank 1 for coverage-shaped questions and instructs the model to decline.

## Documents checked and found NOT defective

- **doc 245** `data coverage: companies, years, and partial-year caveat` —
  **correct**, and directly contradicts 215/225/243. It states "Companies: 478,
  spanning 153 SIC industry…", "Fiscal years loaded: 2020 through 2026",
  "Fiscal year 2026 is PARTIAL: only 24 of 478". This document is *generated
  from the database* by `scripts/build_embeddings.py`; the three defective ones
  are hand-written. That is the root cause: hand-written counts were not
  regenerated when the dataset grew.
- **doc 232** contains "0.253 = 25.3%" — an unrelated percentage, not a count.
- **doc 222** `companies.ticker vs companies.name` lists 25 name→ticker
  mappings, but frames them as "careful mappings", a lookup aid rather than an
  exclusivity claim. Not false. Now covers 25 of 478, so it is *incomplete*
  rather than *wrong*; left unchanged under a minimal-correction rule.
- **docs 240, 244** discuss uncovered companies/years generically with no
  hardcoded counts. Correct as written.
- The remaining 24 documents make no scope, count or coverage claims.

## Correction principle applied

Minimal factual correction only: change the false numbers and the false
counter-examples, keep each document's structure, intent and teaching. No
document rewritten wholesale. Original text remains in git history.
