"""
STEP 6 — Embeddings. Shown, not described.

    python3 06_embeddings.py
    python3 06_embeddings.py "first sentence" "second sentence"   # try your own
    EJO_EMBEDDER=st:LaBSE python3 06_embeddings.py                # a real model

An embedding is a list of numbers that stands for the meaning of a piece of
text. Two texts that mean similar things get lists of numbers that are close
together. That is the whole idea.

Four things, in order:

    1. one real embedding, so you can see it is only numbers
    2. how close every sentence is to every other one
    3. the same numbers drawn as a map, so you can see the groups
    4. the trap — the map can look right for the wrong reason
"""

import argparse
import math
import re

from ejo.embed import cosine, get_embedder

# Three facts from the clinic handbook, each written twice in different words.
# Same meaning, almost no shared vocabulary. A model that understands MEANING
# should put each pair together.
PAIRS = [
    ("phone",
     "Outside opening hours, call the emergency line on 0788 123 456.",
     "When we are shut, dial 0788 123 456 and a nurse picks up."),
    ("fees",
     "A standard consultation costs 3,000 RWF for a patient without insurance.",
     "Uninsured people pay 3,000 RWF to be seen by a doctor."),
    ("jabs",
     "Childhood vaccinations are given every Tuesday morning, 08:00 to 12:00.",
     "We immunise kids on Tuesdays, 08:00 until midday."),
]


def flatten(pairs):
    labels, texts = [], []
    for topic, first, second in pairs:
        labels += [f"{topic}-1", f"{topic}-2"]
        texts += [first, second]
    return labels, texts


# ---------------------------------------------------------------------------
# Drawing the map
# ---------------------------------------------------------------------------

def to_2d(vectors):
    """Squash long lists of numbers down to two each, so we can draw them.

    You do not need the arithmetic to read the map. It finds the two
    directions in which these points differ most and measures every point
    along those two — like picking the camera angle that spreads a group out
    best before taking the photo. The data does not change, only the picture.
    """
    n, dim = len(vectors), len(vectors[0])
    mean = [sum(v[i] for v in vectors) / n for i in range(dim)]
    centred = [[v[i] - mean[i] for i in range(dim)] for v in vectors]
    gram = [[sum(a[i] * b[i] for i in range(dim)) for b in centred] for a in centred]

    def strongest(matrix, seed):
        vec = [math.sin(seed * (i + 1.7)) for i in range(n)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        value = 0.0
        for _ in range(300):
            nxt = [sum(matrix[r][c] * vec[c] for c in range(n)) for r in range(n)]
            value = math.sqrt(sum(x * x for x in nxt))
            if value < 1e-12:
                return 0.0, vec
            vec = [x / value for x in nxt]
        return value, vec

    val1, vec1 = strongest(gram, 1.0)
    residual = [[gram[r][c] - val1 * vec1[r] * vec1[c] for c in range(n)]
                for r in range(n)]
    val2, vec2 = strongest(residual, 2.0)

    sx, sy = math.sqrt(max(val1, 0.0)), math.sqrt(max(val2, 0.0))
    return [(vec1[i] * sx, vec2[i] * sy) for i in range(n)]


def draw(points, labels, width=62, height=13):
    def scale(values, size):
        low, high = min(values), max(values)
        span = high - low
        if span < 1e-9:
            return [size // 2] * len(values)
        return [int((v - low) / span * (size - 1)) for v in values]

    cols = scale([p[0] for p in points], width - 11)
    rows = scale([p[1] for p in points], height - 1)
    grid = [[" "] * width for _ in range(height)]

    for col, row, label in zip(cols, rows, labels):
        row = height - 1 - row                       # y goes up, rows go down
        for attempt in range(height):                # find a free row here
            r = (row + attempt) % height
            end = min(width, col + len(label) + 3)
            if all(grid[r][c] == " " for c in range(col, end)):
                row = r
                break
        grid[row][col] = "*"
        for k, ch in enumerate(label):
            if col + 2 + k < width:
                grid[row][col + 2 + k] = ch

    print("   +" + "-" * width + "+")
    for line in grid:
        print("   |" + "".join(line) + "|")
    print("   +" + "-" * width + "+")


def measure(vectors, labels, topics):
    """Average similarity for 'same meaning' vs 'different fact'."""
    index = {name: i for i, name in enumerate(labels)}
    same = [cosine(vectors[index[f"{t}-1"]], vectors[index[f"{t}-2"]]) for t in topics]
    different = []
    for group in ("1", "2"):
        for a in range(len(topics)):
            for b in range(a + 1, len(topics)):
                different.append(cosine(vectors[index[f"{topics[a]}-{group}"]],
                                        vectors[index[f"{topics[b]}-{group}"]]))
    return (sum(same) / len(same), sum(different) / len(different),
            dict(zip(topics, same)))


# ---------------------------------------------------------------------------

def compare_two(embedder, first: str, second: str) -> None:
    """Score any two sentences. Use this to test a language you speak."""
    a, b = embedder.embed([first, second])
    score = cosine(a, b)
    print("STEP 6 · Embedding similarity")
    print("-" * 68)
    print(f"A: {first}")
    print(f"B: {second}")
    print(f"Similarity: {score:+.3f}")
    print(f"Backend: {embedder.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show how embeddings compare text by distance.",
    )
    parser.add_argument("sentences", nargs="*", help="optional pair of sentences to compare")
    parser.add_argument("--full", action="store_true", help="print the full embedding grid and map")
    args = parser.parse_args()

    embedder = get_embedder()

    if len(args.sentences) >= 2:
        compare_two(embedder, args.sentences[0], args.sentences[1])
        return

    topics = [topic for topic, _, _ in PAIRS]
    labels, texts = flatten(PAIRS)
    vectors = embedder.embed(texts)

    same, different, per_topic = measure(vectors, labels, topics)
    margin = same - different
    stripped = [(t, re.sub(r"[\d,:]+", "", a), re.sub(r"[\d,:]+", "", b))
                for t, a, b in PAIRS]
    s_labels, s_texts = flatten(stripped)
    s_vectors = embedder.embed(s_texts)
    s_same, s_different, s_per_topic = measure(s_vectors, s_labels, topics)
    s_margin = s_same - s_different
    lost = (margin - s_margin) / margin * 100 if margin else 0.0

    if not args.full:
        print("STEP 6 · Embeddings")
        print("-" * 68)
        print(f"Backend: {embedder.name}")
        print(f"Each sentence becomes {embedder.dim} numbers.")
        print()
        print(f"Same-meaning average: {same:.3f}")
        print(f"Different-fact average: {different:.3f}")
        print(f"Signal gap: {margin:.3f}")
        print()
        print(f"After deleting digits, signal gap: {s_margin:.3f}")
        print(f"Signal lost: {lost:.0f}%")
        print()
        print("Answer: embeddings let us search by closeness, but we still")
        print("need to test whether the model is matching meaning or shortcuts.")
        return

    print(f"backend: {embedder.name}")
    print(f"every sentence becomes a list of {embedder.dim} numbers")
    print()
    for name, text in zip(labels, texts):
        print(f"  {name:8s} {text}")
    print()

    # --- 1 -----------------------------------------------------------------
    print("=" * 70)
    print("1 · WHAT ONE EMBEDDING ACTUALLY IS")
    print("=" * 70)
    first = vectors[0]
    biggest = sorted(enumerate(first), key=lambda p: -abs(p[1]))[:8]
    print(f"  {labels[0]} becomes:")
    print("     " + "  ".join(f"{v:+.2f}" for _, v in biggest) + "   … and so on")
    print(f"     ({sum(1 for v in first if abs(v) > 1e-9)} of {embedder.dim} "
          f"numbers are not zero)")
    print()
    print("  That is the whole object. No words are kept inside it. You cannot")
    print("  turn it back into the sentence. It is good for exactly one thing:")
    print("  comparing it against another list of numbers.")
    print()

    # --- 2 -----------------------------------------------------------------
    print("=" * 70)
    print("2 · HOW CLOSE IS EVERY SENTENCE TO EVERY OTHER ONE?")
    print("=" * 70)
    print("  1.00 means the same direction. 0.00 means unrelated.")
    print()
    print("           " + " ".join(f"{n:>8s}" for n in labels))
    for i, name in enumerate(labels):
        row = " ".join(f"{cosine(vectors[i], vectors[j]):>8.2f}"
                       for j in range(len(labels)))
        print(f"  {name:>8s} {row}")
    print()
    print("  Check the diagonal first — every sentence scores 1.00 against")
    print("  itself. If that is ever wrong, something is broken.")
    print()

    # --- 3 -----------------------------------------------------------------
    print("=" * 70)
    print("3 · THE SAME NUMBERS, DRAWN AS A MAP")
    print("=" * 70)
    print("  Close together means similar — according to this model.")
    print()
    draw(to_2d(vectors), labels)
    print()

    print(f"  same meaning, different words   {same:.3f}")
    print(f"  different fact                  {different:.3f}")
    print(f"  the gap between them            {margin:.3f}   <- the signal")
    print()
    if margin > 0:
        print("  The pairs win. Each fact sits closer to its own rewording than")
        print("  to anything else, so this model looks like it is grouping by")
        print("  MEANING. That is what an embedding model is supposed to do.")
    else:
        print("  The pairs lose. This model is not grouping by meaning at all.")
    print()

    # --- 4 · the trap ------------------------------------------------------
    print("=" * 70)
    print("4 · NOW TAKE THE NUMBERS OUT")
    print("=" * 70)
    print("  Look at what each matching pair has in common. '0788 123 456' is")
    print("  in both phone sentences. '3,000 RWF' is in both fees sentences.")
    print("  '08:00' is in both jabs sentences. Those are identical characters,")
    print("  not shared meaning.")
    print()
    print("  So delete every digit and ask again. The MEANING of each sentence")
    print("  is unchanged — only the shared characters are gone.")
    print()

    print(f"  {'':16s} {'with digits':>12s} {'without':>10s} {'change':>9s}")
    for topic in topics:
        before, after = per_topic[topic], s_per_topic[topic]
        drop = (after - before) / before * 100 if before else 0.0
        print(f"  {topic:>14s}   {before:>12.3f} {after:>10.3f} {drop:>8.0f}%")
    print(f"  {'the gap':>14s}   {margin:>12.3f} {s_margin:>10.3f} "
          f"{(s_margin - margin) / margin * 100 if margin else 0:>8.0f}%")
    print()
    draw(to_2d(s_vectors), s_labels)
    print()

    print(f"  Deleting the digits removed {lost:.0f}% of the signal.")
    print()
    print("  Read that carefully. It did not remove any meaning — the sentences")
    print("  still say the same things. What it removed was identical")
    print("  characters appearing in both halves of a pair.")
    print()
    print("  So most of what looked like understanding in section 3 was the")
    print("  phone number matching the phone number. This backend compares")
    print("  characters. It always did. Section 3 just made that hard to see.")
    print()
    print("  THE HABIT THIS IS TEACHING: when something looks like it works,")
    print("  find out why before you believe it.")
    print()

    # --- what to do about it -----------------------------------------------
    print("=" * 70)
    print("WHAT THIS MEANS FOR YOUR PROJECT")
    print("=" * 70)
    print("  The default backend here is deliberately simple so that it needs")
    print("  no download. A real embedding model should hold the pairs")
    print("  together in BOTH tables above. That is the test.")
    print()
    print("  It matters most if your product is not in English. Most embedding")
    print("  models were trained mostly on English. Two checkable facts:")
    print()
    print("    paraphrase-multilingual-MiniLM-L12-v2 publishes a list of 50")
    print("    languages. Kinyarwanda is not on it.")
    print("    LaBSE publishes a list of 109 languages. Kinyarwanda is on it.")
    print()
    print("  A published language list describes what a model was trained on.")
    print("  It is not a measurement of how well it finds YOUR documents. And")
    print("  when a model does not know a language, nothing crashes — no")
    print("  error, no warning. You quietly get the wrong pieces of your")
    print("  document, and the LLM writes a confident answer on top of them.")
    print()
    print("  Test it yourself, in a language you speak:")
    print()
    print('    python3 06_embeddings.py "a sentence" "the same thing, reworded"')
    print('    python3 06_embeddings.py "a sentence" "something unrelated"')
    print()
    print("  You need both numbers. One score on its own tells you nothing.")
    print()
    print("  Models worth testing (each one is a download):")
    print("    paraphrase-multilingual-MiniLM-L12-v2   ~120 MB   384 numbers")
    print("    multilingual-e5-small                   ~470 MB   384 numbers")
    print("    LaBSE                                   ~1.8 GB   768 numbers")
    print()
    print("  Which is best for Kinyarwanda? I do not know, and neither does")
    print("  anyone who has not measured it. Step 11 measures it.")


if __name__ == "__main__":
    main()
