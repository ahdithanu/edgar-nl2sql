"""edgar-nl2sql: natural-language -> SQL RAG system over SEC EDGAR financial data.

Pipeline shape (see app/pipeline.py for the full flow):

    question -> retrieve context (pgvector) -> generate SQL (Claude)
             -> validate (sql_guard) -> execute (read-only Postgres)
             -> self-correct on failure (max 3 attempts) -> plain-English answer
"""
