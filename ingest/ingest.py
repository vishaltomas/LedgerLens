import re
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm

from config import settings
from ingest.parse import DATA_PATH, Parser


class Ingestion:
    def __init__(self, dsn: str = settings.database_url, is_replace: bool = False):
        self.conn = psycopg.connect(dsn)
        self.is_replace = is_replace
        register_vector(self.conn)

    def upsert_document(self, doc: dict) -> None:
        self.conn.execute(
            """
            INSERT INTO documents (doc_name, company, doc_type, fiscal_year, fiscal_quarter,
                                   filing_date)
            VALUES (%(doc_name)s, %(company)s, %(doc_type)s, %(fiscal_year)s,
                    %(fiscal_quarter)s, %(filing_date)s)
            ON CONFLICT (doc_name) DO NOTHING
            """,
            doc,
        )

    def insert_chunks(self, doc_name: str, chunks: list[dict]) -> list[int]:
        ids = []
        with self.conn.cursor() as cur:
            for c in chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (doc_name, page_index, chunk_index, has_table, content)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (doc_name, page_index, chunk_index)
                    DO UPDATE SET content = EXCLUDED.content
                    RETURNING id
                    """,
                    (
                        doc_name,
                        c["page_index"],
                        c["chunk_index"],
                        c.get("has_table", False),
                        c["content"],
                    ),
                )
                ids.append(cur.fetchone()[0])
        return ids

    def insert_embeddings(self, chunk_ids: list[int], vectors: np.ndarray, model_name: str) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO embeddings (chunk_id, model_name, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (chunk_id, model_name) DO UPDATE SET embedding = EXCLUDED.embedding
                """,
                [(cid, model_name, vec) for cid, vec in zip(chunk_ids, vectors)],
            )

    def ingest_document(self, doc: dict, chunks: list[dict], embedder) -> None:
        """One transaction per document: if it fails partway, nothing is written."""
        with self.conn.transaction():
            self.upsert_document(doc)
            ids = self.insert_chunks(doc["doc_name"], chunks)
            vecs = embedder.embed_documents([c["embed_text"] for c in chunks])
            self.insert_embeddings(ids, vecs, settings.embedding_model_name)

    def run(self, files: list[str] | None = None) -> None:
        """Parse PDFs (all when files is None) and ingest them with their FinanceBench metadata."""
        df = pd.read_parquet(Path(DATA_PATH, "financebench.parquet"))
        meta = df.drop_duplicates("doc_name").set_index("doc_name")

        parser = Parser(settings.embedding_model_name)
        parser.extract(files)
        
        for doc_name, chunks in tqdm(parser.chunks.items(), desc="Ingesting", unit="doc"):
            logging.info(f"Ingesting chunks of {doc_name}")
            row = meta.loc[doc_name]
            quarter = re.search(r"Q([1-4])", doc_name)
            doc = {
                "doc_name": doc_name,
                "company": row["company"],
                "doc_type": row["doc_type"].lower(),
                "fiscal_year": int(row["doc_period"]),
                "fiscal_quarter": int(quarter.group(1)) if quarter else None,
                "filing_date": None,
            }
            # adding prefix for better embedding
            prefix = f"{doc['company']} | {doc['doc_type']} | {doc['fiscal_year']}"
            for c in chunks:
                c["embed_text"] = f"{prefix}\n{c['content']}"
            self.ingest_document(doc, chunks, parser.embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FinanceBench PDFs into Postgres")
    parser.add_argument(
        "--file",
        action="append",
        help="PDF file name in data/pdf to ingest, e.g. 3M_2018_10K.pdf (repeatable); "
        "ingests all PDFs when omitted",
    )
    parser.add_argument(
        "--replace",
        action="append",
        help="PDF file name in data/pdf to ingest, e.g. 3M_2018_10K.pdf (repeatable); "
        "ingests all PDFs when omitted",
    )
        
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    Ingestion().run(args.file)
