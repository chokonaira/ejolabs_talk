"""
Splitting documents into chunks.

Why chunk at all? Three reasons, all of which you have already met:

  1. Context window (Session 4). A 40-page document does not fit, and if it
     does, you are paying for all 40 pages on every single question.
  2. Cost. You pay per token, per request. Retrieved text is text you pay for.
  3. Precision. If you retrieve a whole document to answer one question, 39
     of those 40 pages are noise. The model has to find the answer inside the
     noise, and sometimes it does not. This is the "lost in the middle" effect.

The two knobs are chunk size and overlap, and the honest way to tune them is
to print the chunks and read them. There is no correct number.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    """One retrievable piece of text, plus everything needed to cite it.

    Keep the metadata. You need `source` and `section` to show the user where
    an answer came from, and you need `index` and `start`/`end` when you are
    debugging why a particular chunk was retrieved.
    """

    text: str
    source: str            # filename the chunk came from
    section: str           # nearest markdown heading above it
    index: int             # position of this chunk within the document
    start: int             # character offset in the cleaned document
    end: int
    meta: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """Short human-readable tag, used when we number chunks in a prompt."""
        return f"{self.source}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean(text: str) -> str:
    """The 'clean' stage of the pipeline.

    Real ingestion does much more than this: stripping page headers and
    footers, dropping navigation boilerplate, fixing hyphenation from PDF
    extraction, removing duplicate documents. Ours is small on purpose so you
    can see what a cleaning stage *is* rather than what a mature one contains.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Drop HTML/markdown comments: they are notes to the author, not content.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Collapse runs of blank lines to exactly one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _split_on_structure(text: str) -> list[str]:
    """Break text into the smallest pieces we are willing to glue back together.

    Order matters and it is deliberate: paragraph, then sentence, then
    whitespace. We prefer to cut where the author already cut. A chunk that
    ends in the middle of a sentence embeds as half a meaning, and half a
    meaning retrieves badly.
    """
    pieces: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= 400:
            pieces.append(paragraph)
            continue
        # Paragraph is long: fall back to sentence boundaries. The lookbehind
        # keeps the punctuation attached to the sentence it ends.
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= 400:
                pieces.append(sentence)
            else:
                # Still too long: fall back to whitespace. Rare, but a single
                # unbroken 2,000-character line does exist in the wild.
                words, buf = sentence.split(), ""
                for word in words:
                    if len(buf) + len(word) + 1 > 400:
                        pieces.append(buf.strip())
                        buf = word
                    else:
                        buf += " " + word
                if buf.strip():
                    pieces.append(buf.strip())
    return pieces


def split(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a document into overlapping chunks.

    chunk_size — target size in *characters*, not tokens. Characters are what
        you can see and count by eye, which matters when you are teaching or
        debugging. Roughly 500 characters is a paragraph or two.

        Too small and a chunk becomes a fragment with no context: a chunk
        containing only "0788 123 456" is unfindable, because nothing in it
        says whose number it is or when to call it.

        Too big and the one useful line is buried among nine irrelevant ones.

    overlap — how many characters of the previous chunk to repeat at the start
        of the next one. This exists for exactly one reason: a fact that lands
        on a chunk boundary would otherwise be cut in half and exist nowhere
        in one piece. Run 05_chunking.py with overlap=0 to watch that happen.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = clean(text)
    chunks: list[Chunk] = []
    section = ""
    buf = ""
    buf_start = 0
    cursor = 0

    def flush(buf_text: str, start: int, sect: str) -> None:
        if buf_text.strip():
            chunks.append(
                Chunk(
                    text=buf_text.strip(),
                    source=source,
                    section=sect or "(no heading)",
                    index=len(chunks),
                    start=start,
                    end=start + len(buf_text),
                )
            )

    for piece in _split_on_structure(text):
        # Track the nearest heading so every chunk knows which section it is in.
        if piece.startswith("#"):
            # Close the current chunk FIRST, while `section` still names the
            # section that chunk actually came from. Updating `section` before
            # flushing labels every chunk with the heading that follows it,
            # which is an easy off-by-one to write and a miserable one to
            # debug from citations that are all one section too far down.
            if buf:
                flush(buf, buf_start, section)
            section = piece.lstrip("#").strip()
            # A heading alone is not worth retrieving, but it is worth
            # prefixing onto the chunk that follows it, so that chunk carries
            # its own context.
            buf = section + "\n"
            buf_start = cursor
            cursor += len(piece) + 2
            continue

        if len(buf) + len(piece) + 1 > chunk_size and buf.strip():
            flush(buf, buf_start, section)
            # Carry the tail of the chunk we just emitted into the next one.
            tail = buf[-overlap:] if overlap else ""
            buf = tail + " " + piece
            buf_start = cursor - len(tail)
        else:
            buf = (buf + "\n" + piece) if buf else piece

        cursor += len(piece) + 2

    flush(buf, buf_start, section)
    return chunks


def naive_split(
    text: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 0,
) -> list[Chunk]:
    """Fixed-size character windows. The splitter you write first.

    This is not a straw man — it is genuinely the first thing most people
    write, it is what a `text[i:i+n]` loop gives you, and several tutorials
    ship it. It has one virtue (you know exactly how big every chunk is) and
    one serious flaw: it cuts wherever the counter lands, which is usually in
    the middle of a word, a number or a sentence.

    Keep it in the repository so you can run it against `split()` and see the
    difference for yourself. 05_chunking.py does exactly that.
    """
    text = clean(text)
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for start in range(0, len(text), step):
        window = text[start:start + chunk_size]
        if not window.strip():
            continue
        chunks.append(
            Chunk(
                text=window,
                source=source,
                section="(fixed-size window)",
                index=len(chunks),
                start=start,
                end=start + len(window),
            )
        )
    return chunks


def split_file(path: str, **kwargs: Any) -> list[Chunk]:
    """Read a file from disk and split it. `source` defaults to the filename."""
    import os

    with open(path, encoding="utf-8") as handle:
        return split(handle.read(), source=os.path.basename(path), **kwargs)
