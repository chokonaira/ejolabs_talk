"""
A vector store you can read in one sitting.

This is the naive version: keep every vector in a list, and to answer a query
compare it against all of them. That is a linear scan.

    200 chunks      -> instant
    20,000 chunks   -> still fine
    2,000,000 chunks-> unusable, and this is why vector databases exist

A real vector database keeps an approximate-nearest-neighbour index (HNSW is
the common one) so that a lookup does not have to touch every vector. It gives
up a small amount of accuracy for a very large amount of speed. See
09_vector_db_chroma.py and 10_vector_db_pgvector.py.

Read this file first anyway. Once you know that "search" means "sort by cosine
similarity and take the top k", no vector database will ever seem mysterious.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .chunk import Chunk
from .embed import Embedder, cosine


@dataclass
class Hit:
    """One search result: the chunk, and how close it was to the query."""

    chunk: Chunk
    score: float


class VectorStore:
    """Chunks plus their vectors, searchable by cosine similarity."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Embed a batch of chunks and keep them.

        Embed in one batch rather than one call per chunk. With a local model
        that is a large speed difference; with a paid API it is also a cost
        difference.
        """
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Return the k chunks closest in meaning to `query`.

        Note what is happening: the query goes through *the same embedder* as
        the documents did. It has to. Two vectors are only comparable if the
        same model produced them — comparing a MiniLM query vector against
        LaBSE document vectors produces numbers, and the numbers are garbage.
        """
        if not self.chunks:
            return []
        query_vector = self.embedder.embed([query])[0]
        scored = [
            Hit(chunk=chunk, score=cosine(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    # -- persistence --------------------------------------------------------
    # Embedding costs time and sometimes money. Do not recompute vectors every
    # time the script starts. Real ingestion pipelines cache exactly like this,
    # and re-embed only the documents whose content has changed.

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "items": [
                {"chunk": chunk.to_dict(), "vector": vector}
                for chunk, vector in zip(self.chunks, self.vectors)
            ],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["embedder"] != self.embedder.name:
            raise ValueError(
                f"This index was built with {payload['embedder']!r} but you are "
                f"searching it with {self.embedder.name!r}. Vectors from two "
                f"different models are not comparable. Rebuild the index."
            )
        self.chunks = [Chunk(**item["chunk"]) for item in payload["items"]]
        self.vectors = [item["vector"] for item in payload["items"]]


def build_store(doc_paths: list[str], embedder: Embedder, **chunk_kwargs: Any) -> VectorStore:
    """The whole ingestion pipeline in one function.

        load -> clean -> split -> embed -> store

    Each arrow is a place a bug can hide, which is why the stages are named.
    When retrieval returns the wrong chunk, you debug by walking these stages
    in order: was the document loaded, was it cleaned too aggressively, was it
    split somewhere stupid, was it embedded by a model that knows the language.
    """
    from .chunk import split_file

    store = VectorStore(embedder)

    chunks: list[Chunk] = []
    for path in doc_paths:
        chunks.extend(split_file(path, **chunk_kwargs))

    # Some embedders want to see the whole collection before they encode any of
    # it, so they can work out which words are common here and discount them.
    # A downloaded model does not need this; the hashing baseline does.
    if hasattr(embedder, "fit"):
        embedder.fit([chunk.text for chunk in chunks])

    store.add(chunks)
    return store
