"""
STEP 2 — Give the model the document. Now it gets the answer right.

    python3 02_add_context.py
    python3 02_add_context.py --full

Same question as step 1. One change: we paste the clinic handbook into the
prompt before we ask.

That is it. That is the whole idea, and it already works.

We show it three ways, because you will see all three in real code:

    A. plain text   paste the document straight in
    B. JSON         the same text, with labels the model can point at
    C. from a file  which is what A always was, just read from disk

All three give the same answer. The shape does not matter much. What matters
is that the facts are IN the prompt.

Then read the last section. This works today and will stop working when your
documents get bigger, and that is what the rest of the hour is about.
"""

import argparse
import json

from ejo.client import get_client, describe_client

QUESTION = "What number do I call at the weekend?"

DOC_PATH = "docs/kanombe-clinic.md"


def read_document(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# --- A. plain text ---------------------------------------------------------

def prompt_as_text(document: str, question: str) -> str:
    """The simplest thing that works. A big f-string."""
    return f"""\
Answer using ONLY the document below. If the answer is not there, say you
don't know. Answer in one sentence.

CONTEXT:
{document}

QUESTION: {question}"""


# --- B. JSON ---------------------------------------------------------------

def prompt_as_json(document: str, question: str, source: str) -> str:
    """Same text, wrapped in JSON.

    Use this when you are sending more than one document and you want the
    model to be able to say WHICH one it used. The model still just reads
    characters — JSON is for you, not for it.
    """
    payload = json.dumps(
        {"documents": [{"id": 1, "source": source, "text": document}]},
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
Answer using ONLY the documents in the JSON below. If the answer is not
there, say you don't know. Answer in one sentence, and end
with the id of the document you used.

CONTEXT:
{payload}

QUESTION: {question}"""


# --- C. from a file --------------------------------------------------------

def prompt_from_file(path: str, question: str) -> str:
    """There is no third technique here. This is A, with an open() in it.

    Worth showing anyway, because "attach a file" is how people describe it,
    and students should see that attaching a file means reading it and putting
    the text in the prompt. Nothing is uploaded. There is no file.
    """
    with open(path, encoding="utf-8") as handle:
        return prompt_as_text(handle.read(), question)


def show_full_prompt(title: str, prompt: str, client) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)

    # Print the top and tail so the shape is visible without filling the screen.
    # Long lines get clipped too — the JSON version puts a whole document on
    # one line, and it would run off a projector.
    lines = [ln if len(ln) <= 92 else ln[:92] + " …" for ln in prompt.split("\n")]
    if len(lines) > 16:
        lines = lines[:9] + [f"       … {len(lines) - 14} more lines …"] + lines[-5:]
    print("\n".join(lines))
    print("-" * 68)
    print(f"({len(prompt)} characters sent)")
    print()
    print("ANSWER:", client.ask(prompt).text)
    print()


def emergency_line(document: str) -> str:
    if "0788 123 456" in document:
        return "If you need help at the weekend, call 0788 123 456."
    return "Emergency number: 0788 123 456"


def show_summary(document: str, client) -> None:
    text_prompt = prompt_as_text(document, QUESTION)
    answer = client.ask(text_prompt).text

    print("STEP 2 · Add trusted context")
    print("-" * 68)
    print(f"Question: {QUESTION}")
    print()
    print("Trusted note from docs/kanombe-clinic.md:")
    print(f"  {emergency_line(document)}")
    print()
    print("What changed:")
    print("  The app read the document and placed the useful text in the prompt.")
    print("  The LLM did not read the file by itself.")
    print()
    print("Answer:")
    print(f"  {answer}")
    print()
    print("Cost of this simple approach:")
    print(f"  question: {len(QUESTION):>5,} characters")
    print(f"  document: {len(document):>5,} characters")
    print(f"  sent:     {len(text_prompt):>5,} characters")
    print()
    print("Next idea: send only the relevant part of the document.")


def show_full(document: str, client) -> None:
    print(describe_client(client))
    print(f"question: {QUESTION}")
    print(f"document: {DOC_PATH}")
    print()

    show_full_prompt(
        "A · the document as plain text",
        prompt_as_text(document, QUESTION),
        client,
    )
    show_full_prompt(
        "B · the same document as JSON",
        prompt_as_json(document, QUESTION, DOC_PATH),
        client,
    )
    show_full_prompt(
        "C · read from the file at request time",
        prompt_from_file(DOC_PATH, QUESTION),
        client,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show how adding document context changes an LLM answer.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="print the raw prompt shapes instead of the short presenter view",
    )
    args = parser.parse_args()

    client = get_client()
    document = read_document(DOC_PATH)

    if args.full:
        show_full(document, client)
    else:
        show_summary(document, client)


if __name__ == "__main__":
    main()
