"""
STEP 9 — A real vector database, locally. Chroma.

    pip install chromadb
    python3 09_vector_db_chroma.py

Runs without chromadb installed: it prints what it would do and stops. Nothing
in this repository requires a download to be useful.

WHY A VECTOR DATABASE AT ALL

    ejo/store.py compares your question to every chunk, one at a time. That is
    a linear scan:

        200 chunks         instant
        20,000 chunks      fine
        2,000,000 chunks   unusable

    A vector database keeps an index over the vectors — usually HNSW, an
    approximate nearest-neighbour structure — so a lookup does not have to
    touch every vector. Sub-linear instead of linear. You trade a small amount
    of accuracy for a very large amount of speed.

    Definition, plainly: a database whose primary index is over vectors and
    whose primary query is "give me the k nearest".

THE TWO GOTCHAS, BECAUSE EVERYONE HITS BOTH

    1. THE DISTANCE METRIC. Chroma's default space is L2 (squared Euclidean),
       not cosine. On unnormalised vectors those rank differently, and the
       symptom is retrieval that is subtly bad rather than obviously broken.
       Set it explicitly: {"hnsw:space": "cosine"}.

    2. THE DEFAULT EMBEDDING FUNCTION. If you hand Chroma raw text, it embeds
       it for you with its own default model, which is English-only. Your
       Kinyarwanda goes in, English-trained vectors come out, and nothing
       warns you. THE DATABASE DOES NOT UNDERSTAND YOUR LANGUAGE. THE
       EMBEDDING MODEL DOES. Pass your own vectors — as below — or configure
       the embedding function deliberately.

ALTERNATIVES, named once: Qdrant and Weaviate (self-hosted, production),
Pinecone (hosted). And pgvector, which is step 9 and is probably your answer.
"""

import argparse

from ejo.embed import get_embedder
from ejo.store import build_store

DOCS = [
    "docs/kanombe-clinic.md",
    "docs/kanombe-fees.md",
    "docs/kanombe-services.md",
]
QUESTION = "What number do I call at the weekend?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show a local Chroma vector database example.",
    )
    parser.add_argument("--full", action="store_true", help="print implementation details and gotchas")
    args = parser.parse_args()

    try:
        import chromadb
    except ImportError:
        print("STEP 9 · Chroma vector database")
        print("-" * 68)
        print("chromadb is not installed, so nothing was run.")
        print("Install later with: pip install chromadb")
        print()
        print("Answer: a vector database stores chunks + embeddings and returns")
        print("the closest chunks faster than scanning everything.")
        return

    # We chunk and embed with OUR code, then hand Chroma finished vectors.
    # This is the important line of this file: the embedding model is our
    # decision, not the database's default.
    embedder = get_embedder()
    store = build_store(DOCS, embedder, chunk_size=500, overlap=50)
    print(f"embedder: {embedder.name} ({embedder.dim} dims)")
    print(f"chunks  : {len(store.chunks)}")
    print()

    client = chromadb.EphemeralClient()  # in memory; PersistentClient writes to disk
    try:
        client.delete_collection("kanombe")
    except Exception:
        pass

    collection = client.create_collection(
        name="kanombe",
        # GOTCHA 1. Without this you get L2 and quietly different rankings.
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"{c.source}:{c.index}" for c in store.chunks],
        embeddings=store.vectors,          # GOTCHA 2: our vectors, not theirs
        documents=[c.text for c in store.chunks],
        metadatas=[{"source": c.source, "section": c.section} for c in store.chunks],
    )

    result = collection.query(
        query_embeddings=[embedder.embed([QUESTION])[0]],
        n_results=3,
    )

    if not args.full:
        print("STEP 9 · Chroma vector database")
        print("-" * 68)
        print(f"Question: {QUESTION}")
        print(f"Chunks stored: {len(store.chunks)}")
        print()
        print("Top matches:")
        for position in range(len(result["ids"][0])):
            distance = result["distances"][0][position]
            metadata = result["metadatas"][0][position]
            print(f"  [{position + 1}] similarity {1 - distance:.3f}  "
                  f"{metadata['source']} · {metadata['section']}")
        print()
        print("Answer: same retrieval result, but with a vector DB index.")
        return

    print(f"question: {QUESTION}")
    print()
    for position in range(len(result["ids"][0])):
        distance = result["distances"][0][position]
        metadata = result["metadatas"][0][position]
        document = result["documents"][0][position].replace("\n", " ")[:90]
        # Chroma returns DISTANCE, not similarity. With cosine space,
        # similarity = 1 - distance. Getting this backwards sorts your results
        # exactly wrong, and it is a common bug.
        print(f"  [{position + 1}] distance {distance:.3f}  "
              f"(similarity {1 - distance:.3f})")
        print(f"      {metadata['source']} · {metadata['section']}")
        print(f"      {document}…")
        print()

    print("Compare those to step 6. Same chunks, same embedder, same ranking —")
    print("the database changed nothing about the answer, only about how fast")
    print("you get it once there are a million chunks.")
    print()
    print("Re-ingestion, which is the part tutorials skip: when a document")
    print("changes you must DELETE its old chunks before adding the new ones.")
    print("  collection.delete(where={'source': 'kanombe-fees.md'})")
    print("Skip that and you will retrieve last year's fees alongside this")
    print("year's, and the model will confidently pick one.")


if __name__ == "__main__":
    main()
