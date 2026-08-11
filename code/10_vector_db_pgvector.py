"""
STEP 10 — pgvector. Your existing PostgreSQL becomes a vector database.

    python3 10_vector_db_pgvector.py              # prints the SQL, runs nothing
    DATABASE_URL=postgresql://… python3 10_vector_db_pgvector.py

FOR YOUR CAPSTONE, THIS IS ALMOST CERTAINLY THE RIGHT ANSWER.

You already have PostgreSQL from Session 2. You already deploy it. You already
back it up. pgvector is an extension and a column type — one `CREATE
EXTENSION` and your existing database does vector search. No second system to
run, no second thing to secure, no second bill, and your chunks can join
against your ordinary tables in one query, which a dedicated vector database
cannot do.

Reach for Chroma, Qdrant, Weaviate or Pinecone when you have outgrown this.
Most student projects never will.

THE SAME TWO GOTCHAS AS STEP 8

    1. The operator IS the distance metric, and there are three of them:
           <->   L2 (Euclidean)
           <#>   negative inner product
           <=>   COSINE  <-- this one, unless you know why not
       Pick the wrong operator and you get plausible, wrongly-ordered results.

    2. Nothing in the database knows what language your text is. The embedding
       model decides that, and you chose it in ejo/embed.py. Postgres just
       stores the numbers.

ONE MORE THING WORTH KNOWING
    Without an index, pgvector does an exact scan — correct, and fine up to
    tens of thousands of rows. Adding an HNSW index makes it approximate and
    fast. Build the index AFTER bulk loading, not before.
"""

import argparse
import os

from ejo.embed import get_embedder
from ejo.store import build_store

DOCS = [
    "docs/kanombe-clinic.md",
    "docs/kanombe-fees.md",
    "docs/kanombe-services.md",
]
QUESTION = "What number do I call at the weekend?"


SCHEMA = """\
-- Once per database.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. Note that this is an ordinary table: you can add a
-- foreign key to your users table, filter by tenant, join it to anything.
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    source    TEXT   NOT NULL,      -- filename, for citation and for re-ingest
    section   TEXT   NOT NULL,      -- heading, for citation
    position  INT    NOT NULL,      -- chunk index within the document
    content   TEXT   NOT NULL,
    embedding VECTOR({dim}) NOT NULL  -- the dimension is fixed at CREATE time
);

-- Build this AFTER loading your rows, not before.
-- vector_cosine_ops must match the operator you query with (<=>).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

QUERY = """\
-- The entire retrieval step, in SQL.
-- <=> is cosine DISTANCE, so smaller is closer. Similarity is 1 - distance.
SELECT source,
       section,
       content,
       1 - (embedding <=> %(q)s::vector) AS similarity
FROM   chunks
ORDER  BY embedding <=> %(q)s::vector    -- ORDER BY distance, ascending
LIMIT  3;
"""

REINGEST = """\
-- Re-ingestion when a document changes. Delete first, then insert.
-- Forget the DELETE and you will serve last year's fees next to this year's.
BEGIN;
DELETE FROM chunks WHERE source = %(source)s;
-- … INSERT the new chunks here …
COMMIT;
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show how PostgreSQL can store and search vectors with pgvector.",
    )
    parser.add_argument("--full", action="store_true", help="print the SQL schema and query")
    args = parser.parse_args()

    embedder = get_embedder()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("STEP 10 · pgvector")
        print("-" * 68)
        print(f"Embedder: {embedder.name} ({embedder.dim} dimensions)")
        print("DATABASE_URL is not set, so nothing was executed.")
        print()
        print("Answer: pgvector lets your existing PostgreSQL table store")
        print("embeddings and run nearest-chunk search.")
        if args.full:
            print()
            print("SCHEMA")
            print("-" * 68)
            print(SCHEMA.format(dim=embedder.dim))
            print("RETRIEVAL")
            print("-" * 68)
            print(QUERY)
            print("RE-INGESTION")
            print("-" * 68)
            print(REINGEST)
        return

    if args.full:
        print(f"embedder: {embedder.name} ({embedder.dim} dims)")
        print()
        print("=" * 70)
        print("SCHEMA")
        print("=" * 70)
        print(SCHEMA.format(dim=embedder.dim))
        print("=" * 70)
        print("RETRIEVAL")
        print("=" * 70)
        print(QUERY)
        print("=" * 70)
        print("RE-INGESTION")
        print("=" * 70)
        print(REINGEST)

    try:
        import psycopg
    except ImportError:
        print("DATABASE_URL is set but psycopg is not installed.")
        print("  pip install 'psycopg[binary]'")
        return

    store = build_store(DOCS, embedder, chunk_size=500, overlap=50)
    print("=" * 70)
    print(f"RUNNING against {database_url.split('@')[-1]}")
    print("=" * 70)

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            for statement in SCHEMA.format(dim=embedder.dim).split(";"):
                if statement.strip():
                    cursor.execute(statement)

            # Delete before insert, so running this twice does not double up.
            cursor.execute("DELETE FROM chunks WHERE source = ANY(%s)",
                           ([c.source for c in store.chunks],))

            cursor.executemany(
                "INSERT INTO chunks (source, section, position, content, embedding)"
                " VALUES (%s, %s, %s, %s, %s)",
                [
                    (chunk.source, chunk.section, chunk.index, chunk.text, str(vector))
                    for chunk, vector in zip(store.chunks, store.vectors)
                ],
            )
            connection.commit()
            print(f"inserted {len(store.chunks)} chunks")
            print()

            query_vector = str(embedder.embed([QUESTION])[0])
            cursor.execute(QUERY, {"q": query_vector})
            print(f"question: {QUESTION}")
            print()
            for source, section, content, similarity in cursor.fetchall():
                print(f"  {similarity:.3f}  {source} · {section}")
                print(f"         {content.replace(chr(10), ' ')[:88]}…")
                print()

    print("Same ranking as step 6, out of a database you already know how to")
    print("deploy, back up and query.")


if __name__ == "__main__":
    main()
