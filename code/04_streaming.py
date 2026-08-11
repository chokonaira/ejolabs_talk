"""
STEP 4 — Streaming: the same total time, a completely different experience.

    python3 04_streaming.py

BEFORE YOU RUN IT, ASK THE ROOM: which one finishes first?

The answer is neither. Both take the same number of seconds end to end.
Streaming does not make generation faster. It changes TIME TO FIRST TOKEN —
how long the user stares at nothing. Three seconds of blank screen feels
broken; three seconds of text arriving feels fast. That is a Session 3
question — designing a good AI user experience — not a performance question.

AN HONEST NOTE ABOUT EJOCHAT
    The public EjoChat documentation does not describe a streaming mode. So
    this file does not pretend to stream from it. What you get below is:

      (a) a real side-by-side timing demo against a local generator, so you
          can see and feel the difference, and
      (b) the actual FastAPI code you would write, which is the part you keep.

    If your provider does not stream, you can still build the streamed UI —
    you simply will not get the latency win until the upstream supports it.
    Do not fake it by chopping a finished reply into pieces and adding sleeps:
    that is slower than just showing the answer, and it lies to your user.
"""

import argparse
import sys
import time

TOKENS = (
    "At the weekend , call the emergency line on 0788 123 456 . "
    "That line is answered by the duty nurse , 24 hours a day ."
).split(" ")

TOKEN_DELAY = 0.12  # what a real model's per-token latency feels like


def generate():
    """Stand-in for a provider that streams. Yields one token at a time."""
    for token in TOKENS:
        time.sleep(TOKEN_DELAY)
        yield token + " "


def blocking() -> None:
    """Wait for the whole answer, then print it. The default, and it is fine
    for a script — it is only wrong when a human is watching a screen."""
    started = time.perf_counter()
    text = "".join(generate())          # nothing reaches the user until this returns
    first_token_at = time.perf_counter() - started
    print(text)
    total = time.perf_counter() - started
    print(f"\n  time to first token: {first_token_at:5.2f}s")
    print(f"  total:               {total:5.2f}s")


def streamed() -> None:
    """Print each token as it arrives.

    The only real change is that we stopped calling "".join() on the generator
    and started consuming it. flush=True matters: without it Python buffers
    the output and you get the blocking behaviour back by accident.
    """
    started = time.perf_counter()
    first_token_at = None
    for token in generate():
        if first_token_at is None:
            first_token_at = time.perf_counter() - started
        sys.stdout.write(token)
        sys.stdout.flush()
    total = time.perf_counter() - started
    print(f"\n  time to first token: {first_token_at:5.2f}s")
    print(f"  total:               {total:5.2f}s")


FASTAPI_EXAMPLE = '''\
# In your capstone backend. This is the whole of it.

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/ask")
def ask(q: str):
    def event_stream():
        for token in your_model_stream(q):     # any generator of strings
            yield f"data: {token}\\n\\n"        # SSE frame: "data: ", then two newlines
        yield "data: [DONE]\\n\\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# And in the browser, the client half is four lines:
#   const es = new EventSource("/ask?q=" + encodeURIComponent(question));
#   es.onmessage = (e) => {
#     if (e.data === "[DONE]") return es.close();
#     output.textContent += e.data;
#   };
'''


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare blocking output with streamed output.",
    )
    parser.add_argument("--full", action="store_true", help="also print the FastAPI example")
    args = parser.parse_args()

    print("BLOCKING — the user sees nothing until it is finished")
    print("-" * 68)
    blocking()
    print()

    print("STREAMED — the user starts reading immediately")
    print("-" * 68)
    streamed()
    print()

    print("Same total. Different product.")
    if args.full:
        print()
        print("=" * 68)
        print("The FastAPI half, which is what you actually keep:")
        print("=" * 68)
        print(FASTAPI_EXAMPLE)


if __name__ == "__main__":
    main()
