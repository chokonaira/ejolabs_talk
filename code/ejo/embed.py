"""
Embeddings — turning text into a list of numbers that stands for its meaning.

WHAT AN EMBEDDING IS
    A fixed-length list of numbers produced by a model, such that two texts
    with similar meaning produce two lists that point in a similar direction.

WHAT AN EMBEDDING IS NOT
    Not a summary. Not a compression you can reverse — you cannot get the text
    back out. Not the same thing as the LLM: it is a separate model, called
    separately, paid for separately, and it can be good at your language while
    the LLM is bad at it, or the other way round.

THE INTUITION, AS A LADDER
    1 number  -> a point on a line.
    2 numbers -> a point on a page.
    3 numbers -> a point in this room.
    384       -> a space nobody can draw. But "which two points are closest"
                 is computed exactly the same way it is in the room.

MEASURING CLOSENESS — COSINE SIMILARITY
    cos(a, b) = dot(a, b) / (norm(a) * norm(b))
    Range -1 to 1. 1 means the two texts point the same direction in meaning,
    0 means unrelated, -1 means opposite. On normalised vectors the divisor is
    1, so cosine similarity is just the dot product — which is why almost every
    vector database normalises on the way in.

A WARNING THAT MATTERS FOR KINYARWANDA
    Most embedding models were trained overwhelmingly on English. Several
    "multilingual" models are built on XLM-R. The published language list for
    `paraphrase-multilingual-MiniLM-L12-v2` names 50 languages and Kinyarwanda
    is not among them. LaBSE's published list of 109 languages does include
    `rw`. Neither fact tells you how well either model will retrieve *your*
    documents — a published language list is a claim about training data, not
    a measurement of retrieval quality.

    When an embedding model does not understand a language, nothing crashes.
    There is no error and no warning. It quietly returns the wrong chunks, and
    the LLM then writes a confident wrong answer on top of plumbing that is
    working perfectly.

    So: do not take a recommendation, including this one. Measure. That is
    what 11_evaluate.py is for, and why the `random` backend below exists —
    a model that cannot beat random on your corpus does not know your language.
"""

from __future__ import annotations

import hashlib
import math
import random as _random
import re
from typing import Protocol


# ---------------------------------------------------------------------------
# Cosine similarity — the whole of it
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    """cos(a, b) = dot(a, b) / (|a| * |b|).

    Written out in full rather than imported, because it is three lines and
    every student should see that there is no magic in it.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# The backend interface
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    """Anything that can turn a list of strings into a list of vectors."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Backend 1: hashing — the zero-download fallback
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """A lexical vectoriser built from the standard library alone.

    BE HONEST ABOUT WHAT THIS IS. It matches characters, not meaning. Two
    sentences that say the same thing in different words score near zero. It
    is not a semantic model and calling it an "embedding model" would be a lie.

    It is here for three good reasons:
      1. It downloads nothing, so the pipeline runs on a dead wifi connection.
      2. It is a real baseline. Character n-grams are a genuinely strong
         retrieval method on a small corpus, and on a morphologically rich
         language like Kinyarwanda they handle prefixes and suffixes better
         than word matching does.
      3. It gives 11_evaluate.py something to beat. If a 500 MB multilingual
         model cannot beat 80 lines of hashing on your corpus, that is the
         single most useful thing you will learn all week.

    How it works: take the words and the overlapping 4-character sequences of
    the text, hash each one into a bucket, count them, then normalise. Similar
    text shares buckets. That is all.
    """

    def __init__(self, dim: int = 512, char_ngram: int = 4) -> None:
        self.dim = dim
        self.char_ngram = char_ngram
        self.name = f"hashing(dim={dim},ngram={char_ngram})"
        self._idf: list[float] | None = None

    def fit(self, texts: list[str]) -> None:
        """Learn which features are common in this collection, and discount them.

        Without this, "the" and "for" count as much as "malaria". In English
        that is fatal: every chunk contains the common words, so every chunk
        looks equally similar to every question and the ranking goes flat.

        The fix is the oldest trick in information retrieval. Count how many
        chunks each feature appears in, and weight it by the inverse — a
        feature in every chunk tells you nothing, a feature in one chunk tells
        you a great deal. That is IDF, inverse document frequency.

        Call this once, on your chunks, before embedding anything. The query
        must then be embedded by the same fitted object, or the weights will
        not match.
        """
        n = len(texts)
        if n == 0:
            return
        seen_in = [0] * self.dim
        for text in texts:
            for bucket in {self._bucket(f) for f in self._features(text)}:
                seen_in[bucket] += 1
        # +1 top and bottom so a feature in every chunk gets a small positive
        # weight rather than exactly zero, and nothing divides by zero.
        self._idf = [math.log((n + 1) / (count + 1)) + 1.0 for count in seen_in]

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        words = _WORD_RE.findall(text)
        feats = [f"w:{w}" for w in words]
        # Character n-grams over the whitespace-collapsed string. These are
        # what make the backend work across word-form changes.
        flat = " ".join(words)
        n = self.char_ngram
        feats += [f"c:{flat[i:i + n]}" for i in range(max(0, len(flat) - n + 1))]
        return feats

    def _bucket(self, feature: str) -> int:
        # Python's built-in hash() is salted per process, so the same text
        # would land in different buckets on different runs. Use a stable
        # hash instead. This is a real bug students hit; better to meet it here.
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for feature in self._features(text):
                vec[self._bucket(feature)] += 1.0
            # Sublinear scaling: a word appearing 20 times is not 20 times as
            # important as a word appearing once.
            vec = [1.0 + math.log(v) if v > 0 else 0.0 for v in vec]
            # Then discount features that appear everywhere, if we have been
            # shown the collection. See fit().
            if self._idf is not None:
                vec = [v * w for v, w in zip(vec, self._idf)]
            out.append(_l2_normalise(vec))
        return out


# ---------------------------------------------------------------------------
# Backend 2: random — the null model
# ---------------------------------------------------------------------------

class RandomEmbedder:
    """Deterministic nonsense. The floor that every real model must clear.

    Include this in every evaluation you run. It converts "our retrieval seems
    to work" into a number you can compare against, and it is the fastest way
    to catch a broken pipeline: if your real model scores the same as this one,
    something is wired wrong — or the model does not know your language.
    """

    def __init__(self, dim: int = 384, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        self.name = f"random(dim={dim})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            # Seed from the text so the same text always gets the same vector.
            # Random, but not different every time you call it.
            rng = _random.Random(f"{self.seed}:{text}")
            out.append(_l2_normalise([rng.gauss(0, 1) for _ in range(self.dim)]))
        return out


# ---------------------------------------------------------------------------
# Backend 3: sentence-transformers — a real embedding model
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """Wraps a real model from the sentence-transformers library.

    This one downloads weights: roughly 120 MB for MiniLM, roughly 1.8 GB for
    LaBSE. Do that at home on a good connection, not in the lecture theatre.
    `python3 setup_check.py --download <model>` will do it for you.

    Candidates worth putting in your evaluation:
        paraphrase-multilingual-MiniLM-L12-v2   ~120 MB, 384 dims
        multilingual-e5-small                   ~470 MB, 384 dims
        LaBSE                                   ~1.8 GB, 768 dims

    Which is best for Kinyarwanda? I do not know, and neither does anyone who
    has not measured it on a corpus like yours. Run 11_evaluate.py.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "  pip install sentence-transformers\n"
                "Or use the hashing backend, which needs no download:\n"
                "  EJO_EMBEDDER=hashing python3 08_rag.py"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


# ---------------------------------------------------------------------------
# Choosing a backend
# ---------------------------------------------------------------------------

def get_embedder(spec: str | None = None) -> Embedder:
    """Build an embedder from a short string, so backends are swappable.

        get_embedder("hashing")                        zero download
        get_embedder("random")                         the null model
        get_embedder("st:LaBSE")                       a real model
        get_embedder()                                 reads $EJO_EMBEDDER,
                                                       defaults to hashing

    Defaulting to `hashing` is a deliberate choice: every script in this
    repository must run for a student with no internet and no API key.
    """
    import os

    spec = spec or os.environ.get("EJO_EMBEDDER", "hashing")
    spec = spec.strip()

    if spec == "hashing":
        return HashingEmbedder()
    if spec == "random":
        return RandomEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    raise ValueError(
        f"Unknown embedder {spec!r}. Use 'hashing', 'random', or 'st:<model-name>'."
    )
