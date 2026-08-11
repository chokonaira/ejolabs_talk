"""
STEP 5 — Chunking, and the bug that overlap exists to prevent.

    python3 05_chunking.py

The ingestion pipeline, named as a pipeline so you can debug it one stage at
a time:

    load  ->  clean  ->  split into chunks  ->  embed  ->  store  ->  retrieve

This script is the "split" stage on its own, with the output printed. The two
knobs are chunk size and overlap. There are no correct values. There is only
looking at the chunks and deciding whether you would want to read them.

The script PROVES the boundary bug rather than asserting it: it searches for
a chunk size at which the emergency number gets cut in half, prints
both halves, and then turns overlap on and shows the number come back.
"""

import argparse

from ejo.chunk import clean, naive_split, split_file

DOC = "docs/kanombe-clinic.md"
NUMBER = "0788 123 456"

# The emergency line. This phrase occurs exactly once in the document, which
# is what makes it a fair test: a fact that appears twice in your documents is
# protected by accident, not by your chunker.
TARGET = "call the emergency number: 0788 123 456"


def flat(text: str) -> str:
    """Collapse every run of whitespace to a single space.

    The documents are hard-wrapped at about 78 characters, so a phrase we look
    for often has a line break in the middle of it. We care whether the FACT
    survived the chunker, not how the file happens to be wrapped.
    """
    return " ".join(text.split())


def survives(chunks, needle: str = TARGET) -> bool:
    """Is `needle` present, unbroken, in at least one chunk?"""
    return any(flat(needle) in flat(chunk.text) for chunk in chunks)


def load() -> str:
    with open(DOC, encoding="utf-8") as handle:
        return clean(handle.read())


def show(chunks, limit: int = 3) -> None:
    for chunk in chunks[:limit]:
        head = chunk.text.replace("\n", " ")
        head = head[:105] + ("…" if len(chunk.text) > 105 else "")
        print(f"  [{chunk.index}] {chunk.source} · {chunk.section}")
        print(f"      {len(chunk.text):3d} chars | {head}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show why documents are split into chunks before retrieval.",
    )
    parser.add_argument("--full", action="store_true", help="print the detailed chunking walkthrough")
    args = parser.parse_args()

    text = load()

    # --- 1. Sensible settings, splitting on structure ----------------------
    chunks = split_file(DOC, chunk_size=500, overlap=50)
    breaking = next(
        (size for size in range(250, 801)
         if not survives(naive_split(text, DOC, size, overlap=0))),
        None,
    )

    if not args.full:
        print("STEP 5 · Chunking")
        print("-" * 68)
        print(f"Document: {DOC}")
        print(f"Normal split: {len(chunks)} chunks, chunk_size=500, overlap=50")
        print(f"Emergency fact survives: {survives(chunks)}")
        print()
        if breaking is not None:
            broken = naive_split(text, DOC, breaking, overlap=0)
            fixed = naive_split(text, DOC, breaking, overlap=50)
            print(f"Boundary test at chunk_size={breaking}:")
            print(f"  no overlap:    survives = {survives(broken)}")
            print(f"  with overlap:  survives = {survives(fixed)}")
        print()
        print("Answer: overlap keeps important facts from being split in half.")
        return

    print(f"{DOC} — split on structure, chunk_size=500, overlap=50")
    print(f"  {len(chunks)} chunks, average "
          f"{sum(len(c.text) for c in chunks) // len(chunks)} characters")
    print()
    show(chunks)
    print()
    print("  Every chunk carries its source and its section. You need both to")
    print("  cite an answer, and you need them again when you are working out")
    print("  which chunk produced a wrong one.")
    print()

    # --- 2. Too small ------------------------------------------------------
    print("=" * 68)
    print("TOO SMALL — chunk_size=120")
    print("=" * 68)
    show(split_file(DOC, chunk_size=120, overlap=0), limit=4)
    print()
    print("  Read these as the model will. A chunk saying only '0788 123 456'")
    print("  does not say whose number it is or when to call it. It is a")
    print("  fragment: it will not be retrieved, and if it is, it will not help.")
    print()

    # --- 3. The boundary bug, with a fixed-size splitter -------------------
    print("=" * 68)
    print("THE BOUNDARY BUG — fixed-size windows, overlap=0")
    print("=" * 68)
    print("This is the splitter almost everyone writes first: text[i:i+n] in a")
    print("loop. It cuts wherever the counter lands.")
    print()
    print(f"  The fact we are trying to keep: {TARGET!r}")
    print()

    # Search upward from a realistic size, so the number on screen is one a
    # student might plausibly have chosen.
    if breaking is None:
        print("  No size in 250–800 splits it on this document. Point TARGET at")
        print("  another fact and run again.")
    else:
        broken = naive_split(text, DOC, breaking, overlap=0)
        print(f"  chunk_size={breaking}, overlap=0  ->  survives? "
              f"{survives(broken)}")
        print()
        # Find the pair of chunks the fact fell between and show the seam.
        for i in range(1, len(broken)):
            if "123 456" in broken[i].text and "123 456" not in broken[i - 1].text:
                if TARGET in broken[i].text:
                    continue
                print(f"  chunk [{i - 1}] ends   …{broken[i - 1].text[-42:]!r}")
                print(f"  chunk [{i}] starts …{broken[i].text[:42]!r}")
                break
        print()
        print("  It is in two pieces. Part is at the end of one chunk, part at")
        print("  the start of the next, and the fact exists nowhere in one")
        print("  piece. No embedding model and no clever query can find it.")
        print("  Retrieval is working perfectly and the answer is unreachable.")
        print()

        fixed = naive_split(text, DOC, breaking, overlap=50)
        print(f"  chunk_size={breaking}, overlap=50 ->  survives? "
              f"{survives(fixed)}")
        print()
        print("  That is the whole reason overlap exists: repeat the tail of")
        print("  each chunk at the head of the next, so a fact that lands on a")
        print("  boundary still survives whole somewhere.")
    print()

    # A warning worth 30 seconds, because it decides whether testing this on
    # your own documents tells you anything at all.
    print(f"  When you test this on your own documents, pick a fact that")
    print(f"  appears EXACTLY ONCE. '{NUMBER}' appears {text.count(NUMBER)} "
          f"time here, which is")
    print("  what makes it a fair test. If a fact is written twice — in a")
    print("  summary and again in the body, say — then cutting one copy leaves")
    print("  the other whole, and your chunker looks fine at every size.")
    print("  It is not fine. It is lucky. Redundancy hides chunking bugs.")
    print()

    # --- 4. Why structure beats a character counter ------------------------
    print("=" * 68)
    print("WHY WE SPLIT ON STRUCTURE INSTEAD")
    print("=" * 68)
    print("  Does the fact survive, at overlap=0, under each splitter?")
    print()
    print(f"  {'size':>6}  {'fixed-size':>12}  {'on structure':>14}")
    for size in (200, 300, 400, 500, 600):
        a = survives(naive_split(text, DOC, size, overlap=0))
        b = survives(split_file(DOC, chunk_size=size, overlap=0))
        print(f"  {size:>6}  {str(a):>12}  {str(b):>14}")
    print()
    print("  Splitting on paragraph, then sentence, then whitespace means we")
    print("  cut where the author already cut. A chunk that ends mid-sentence")
    print("  embeds as half a meaning, and half a meaning retrieves badly.")
    print()
    print("  Keep the overlap anyway. Structure protects a fact inside one")
    print("  sentence; it does not protect a fact that needs the sentence")
    print("  before it to make sense.")
    print()
    print("=" * 68)
    print("What real ingestion also has to handle, which this does not:")
    print("  PDFs and scans (OCR), tables, headers and footers, duplicate")
    print("  documents, and re-ingestion when a document changes — you must")
    print("  delete the old chunks first, or you will retrieve last year's")
    print("  answer next to this year's and never know which one you got.")


if __name__ == "__main__":
    main()
