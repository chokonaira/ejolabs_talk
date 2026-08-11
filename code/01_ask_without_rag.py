"""
STEP 1 — Call an LLM from your own code, and watch it lie to you.

    python3 01_ask_without_rag.py

This is an HTTP POST with a JSON body. Nothing more. It is the same `requests`
you used in Session 2, the same status codes you saw in Postman, and the key
goes in a header. If you can call any REST API, you can call this one.

The interesting part is not the call. It is the answer.

We ask about a small clinic in Kanombe. The model has never seen that clinic —
it was not in the training data, and it should not have been. It does not say
"I have never heard of this place". It produces a well-formed, polite,
confident, invented phone number.

Nothing in the prompt gave it permission to fail, so it does not fail. It
fabricates. That is hallucination, from Session 4, in one screen of output.
The rest of this hour is the engineering answer to it.
"""

import argparse

from ejo.client import get_client, describe_client
from ejo.prompt import build_plain_prompt

QUESTION = "What number do I call at the weekend?"
# The ground truth, which lives in docs/kanombe-clinic.md and which the model
# has no way of knowing:
TRUTH = "0788 123 456"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask without document context and show the answer.",
    )
    parser.add_argument("--full", action="store_true", help="print the prompt and teaching notes")
    args = parser.parse_args()

    client = get_client()

    prompt = build_plain_prompt(QUESTION)
    reply = client.ask(prompt)

    if not args.full:
        print("STEP 1 · Ask without trusted context")
        print("-" * 68)
        print(f"Question: {QUESTION}")
        print()
        print("Answer:")
        print(f"  {reply.text}")
        print()
        print(f"Correct answer from the document: {TRUTH}")
        print("Takeaway: the answer sounds confident, but it is not grounded.")
        return

    print(describe_client(client))
    print()
    print("PROMPT SENT")
    print("-" * 68)
    print(prompt)
    print("-" * 68)
    print()

    print("ANSWER")
    print("-" * 68)
    print(reply.text)
    print("-" * 68)
    print()

    print(f"The real number, from docs/kanombe-clinic.md: {TRUTH}")
    if TRUTH.replace(" ", "") in reply.text.replace(" ", ""):
        print("It happened to be right. Ask it again, or ask about your own")
        print("organisation instead — the point survives the coincidence.")
    else:
        print("The answer is wrong, and notice HOW it is wrong: fluent, polite,")
        print("formatted like a fact. The failure mode is confidence, not silence.")
        print()
        print("Four reasons this happens, and none of them are fixable by prompting harder:")
        print("  1. Training cutoff       — it cannot know anything after a certain date.")
        print("  2. Private data          — this clinic's handbook was never public.")
        print("  3. No permission to fail — we never told it 'you may say I don't know'.")
        print("  4. Facts change          — even a memorised fact goes stale and stays stale.")


if __name__ == "__main__":
    main()
