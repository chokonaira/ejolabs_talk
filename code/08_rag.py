"""
STEP 8 — The whole thing. Retrieval + Augmentation + Generation.

    python3 08_rag.py
    python3 08_rag.py "Inkingo z'abana zitangwa ryari?"
    python3 08_rag.py "Ni ryari ivuriro ryafunguwe bwa mbere?"   # not in the corpus

This is the same question we asked in step 1, asked twice more: once with no
context, once with retrieved context. Same model, same question, two answers.
The only thing that changed is what went into the prompt.

    RAG = find the relevant pages, paste them into the prompt, then ask.

It is not a model, not a library and not a product. It is an architecture — a
pattern for what you put in the prompt. Nobody installs RAG.

The most important line of output in this file is the assembled prompt. Read
it on screen. Everything students find mysterious about RAG stops being
mysterious the moment they see the string that was actually sent.
"""

import argparse

from ejo.client import get_client, describe_client
from ejo.embed import get_embedder
from ejo.prompt import build_plain_prompt, build_rag_prompt
from ejo.store import build_store

DOCS = [
    "docs/kanombe-clinic.md",
    "docs/kanombe-fees.md",
    "docs/kanombe-services.md",
]

DEFAULT_QUESTION = "What number do I call at the weekend?"
TOP_K = 3


def rule(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval-augmented generation for one question.",
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--full", action="store_true", help="print the raw prompt and detailed walkthrough")
    args = parser.parse_args()
    question = args.question

    client = get_client()
    embedder = get_embedder()

    store = build_store(DOCS, embedder, chunk_size=500, overlap=50)
    hits = store.search(question, k=TOP_K)
    plain_prompt = build_plain_prompt(question)
    rag_prompt = build_rag_prompt(question, hits)
    plain_answer = client.ask(plain_prompt).text
    reply = client.ask(rag_prompt)

    if not args.full:
        print("STEP 8 · RAG")
        print("-" * 68)
        print(f"Question: {question}")
        print()
        print("Without document context:")
        print(f"  {plain_answer}")
        print()
        print("With retrieved context:")
        print(f"  {reply.text}")
        print()
        print("Sources:")
        for position, hit in enumerate(hits, start=1):
            print(f"  [{position}] {hit.score:.3f}  {hit.chunk.source} · {hit.chunk.section}")
        print()
        print("Answer: RAG finds relevant chunks, puts them in the prompt,")
        print("then asks the model to answer from those chunks.")
        return

    print(describe_client(client))
    print(f"embedder: {embedder.name}")
    print(f"question: {question}")

    # ---------------------------------------------------------------- no RAG
    rule("A · NO RETRIEVAL — the model answers from memory")
    print(plain_prompt)
    print("-" * 68)
    print(plain_answer)

    # ------------------------------------------------------------ retrieval
    rule("B · RETRIEVAL — search our own documents. No AI yet.")
    print(f"{len(store.chunks)} chunks indexed. Top {TOP_K}:")
    for position, hit in enumerate(hits, start=1):
        print(f"  [{position}] {hit.score:.3f}  {hit.chunk.source} · {hit.chunk.section}")

    # ----------------------------------------------------------- the prompt
    rule("C · AUGMENTATION — the prompt we are about to send, in full")
    print(rag_prompt)
    print("-" * 68)
    print(f"({len(rag_prompt)} characters. Every one of them is billed, every "
          f"time you ask.)")
    print()
    print("The four parts every RAG prompt needs:")
    print("  1. INSTRUCTION  — including permission to say 'I don't know'")
    print("  2. CONTEXT      — numbered, each item labelled with its source")
    print("  3. QUESTION     — the user's words, unmodified")
    print("  4. OUTPUT RULE  — language, length, cite the item you used")
    print()
    print("Note where the output rule physically sits: at the TOP, with the")
    print("instruction. Those are four things the prompt must contain, not four")
    print("blocks in that order on the wire. The one hard ordering rule is that")
    print("the QUESTION goes LAST — instructions top, context middle, question")
    print("bottom.")

    # -------------------------------------------------------- the generation
    rule("D · GENERATION — the model reads that prompt and writes the answer")
    print(reply.text)
    if reply.meta:
        print()
        print(f"  [mock internals: {reply.meta}]")

    # ------------------------------------------------------------- sources
    rule("E · SOURCES — always show these")
    for position, hit in enumerate(hits, start=1):
        print(f"  [{position}] {hit.chunk.source} · {hit.chunk.section} "
              f"(similarity {hit.score:.3f})")
    print()
    print("This is one line of code and it is how your user catches you being")
    print("wrong. Put it in your capstone. RAG produces confident answers that")
    print("are grounded in a retrieved chunk — which is more convincing than a")
    print("plain guess, and therefore more dangerous when the chunk is wrong.")
    print()
    print("Two more questions worth running, each showing something different:")
    print()
    print('  python3 08_rag.py "How much does an X-ray cost?"')
    print("     The documents never mention X-rays. Nothing scores well")
    print("     enough, so it refuses — that is the 'say you don't know' line")
    print("     in the prompt doing its job.")
    print()
    print('  python3 08_rag.py "When did the clinic first open?"')
    print("     Not in any document either. But the word 'open' is all over")
    print("     the opening-hours section, so retrieval hands that back and")
    print("     the answer comes out confident, grounded, cited and wrong.")
    print("     No prompt wording fixes this. The bug is in retrieval, not")
    print("     in generation.")


if __name__ == "__main__":
    main()
