"""
ejo — the small shared library behind the numbered demo scripts.

Four modules, none of them clever:

    ejo.chunk   split a document into overlapping chunks that keep their metadata
    ejo.embed   turn text into a vector; several swappable backends
    ejo.store   hold vectors in memory, find the k nearest by cosine similarity
    ejo.client  talk to the EjoChat API, or to an offline stand-in

Everything here runs with the Python standard library alone. The optional
extras (sentence-transformers, chromadb, psycopg) are only needed for the
scripts that say so at the top of the file.
"""

__all__ = ["chunk", "embed", "store", "client"]
