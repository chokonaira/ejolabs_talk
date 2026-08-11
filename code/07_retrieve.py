"""
STEP 7 — Retrieval. The R in RAG, entirely on its own.

    python3 07_retrieve.py
    python3 07_retrieve.py "Inkingo z'abana zitangwa ryari?"

No AI generation happens in this file. Nothing is written, nothing is
summarised. This is search: embed the question, compare it against every
chunk, sort by cosine similarity, take the top k.

That is the whole of retrieval. If you can read this file, you can debug the
half of RAG that goes wrong most often.

ASK THE ROOM BEFORE YOU PRESS ENTER: three documents, about twenty chunks.
Which chunk do you think wins, and by how much?
"""

import argparse

from ejo.embed import get_embedder
from ejo.store import build_store

DOCS = [
    "docs/kanombe-clinic.md",
    "docs/kanombe-fees.md",
    "docs/kanombe-services.md",
]

DEFAULT_QUESTION = "What number do I call at the weekend?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve the closest document chunks for a question.",
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--full", action="store_true", help="print full retrieval debugging notes")
    args = parser.parse_args()
    question = args.question

    embedder = get_embedder()
    store = build_store(DOCS, embedder, chunk_size=500, overlap=50)

    hits = store.search(question, k=5)

    if not args.full:
        print("STEP 7 · Retrieval")
        print("-" * 68)
        print(f"Question: {question}")
        print(f"Indexed: {len(store.chunks)} chunks from {len(DOCS)} documents")
        print()
        print("Top matches:")
        for position, hit in enumerate(hits[:3], start=1):
            print(f"  [{position}] {hit.score:.3f}  {hit.chunk.source} · {hit.chunk.section}")
        print()
        print("Answer: retrieval is search. No LLM has written anything yet.")
        return

    print(f"backend : {embedder.name}")
    print(f"indexed : {len(store.chunks)} chunks from {len(DOCS)} documents")
    print(f"question: {question}")
    print()

    print("TOP 5 BY COSINE SIMILARITY")
    print("-" * 68)
    for position, hit in enumerate(hits, start=1):
        text = hit.chunk.text.replace("\n", " ")
        text = text[:150] + ("…" if len(hit.chunk.text) > 150 else "")
        bar = "█" * max(0, int(round(hit.score * 40)))
        print(f"[{position}] {hit.score:.3f} {bar}")
        print(f"    {hit.chunk.source} · {hit.chunk.section}")
        print(f"    {text}")
        print()

    print("-" * 68)
    print("Three things to notice, and they are the whole of retrieval debugging:")
    print()
    print("1. THE GAP MATTERS MORE THAN THE SCORE. If [1] and [2] are 0.61 and")
    print("   0.60, retrieval is guessing. A clear gap means it is confident.")
    print()
    print("2. AN ABSOLUTE SCORE MEANS NOTHING ON ITS OWN. 0.4 is good for one")
    print("   model and bad for another. Never hard-code a similarity")
    print("   threshold you have not measured on your own corpus.")
    print()
    print("3. THIS IS A LINEAR SCAN. We compared the question to every single")
    print("   chunk. Instant at 20, fine at 20,000, hopeless at 2 million.")
    print("   That is the entire reason vector databases exist — see step 8.")


if __name__ == "__main__":
    main()
