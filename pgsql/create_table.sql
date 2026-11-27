CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    doc_name        text PRIMARY KEY,   -- joins to FinanceBench
    company         text NOT NULL,
    doc_type        text NOT NULL,      -- 10k, 10q, 8k, earnings
    fiscal_year     int  NOT NULL,
    fiscal_quarter  int,                -- NULL for 10-K and 8-K
    filing_date     date                -- separates 8-Ks from the same year
);

CREATE TABLE chunks (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_name     text NOT NULL REFERENCES documents ON DELETE CASCADE,
    page_index   int  NOT NULL,         -- 0-based, like evidence_page_num
    chunk_index  int  NOT NULL,
    has_table    boolean NOT NULL DEFAULT false,
    content      text NOT NULL,         -- shown to the LLM; the embedding prefix comes from documents
    search_text  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (doc_name, page_index, chunk_index)
);

CREATE INDEX ON chunks USING gin (search_text);

-- One row per chunk per embedding model, so models can be compared side by side
CREATE TABLE embeddings (
    chunk_id     bigint NOT NULL REFERENCES chunks ON DELETE CASCADE,
    model_name   text   NOT NULL,       -- e.g. 'BAAI/bge-base-en-v1.5'
    embedding    vector(768) NOT NULL,  -- normalized; must match the model's dimension
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model_name)
);
