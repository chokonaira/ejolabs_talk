"""
STEP 3 — Getting structured output, so your program can act on the answer.

    python3 03_structured_output.py

THE PROBLEM
    An `if` statement cannot read a paragraph. If the model replies
    "this sounds quite serious, you should probably call someone", your code
    cannot route it, store it, or count it. You need fields.

THE METHOD
    1. Ask for JSON, and show an example of the exact shape you want.
    2. Enumerate the allowed values. "urgency" is not a free-text field; it is
       one of three strings, listed.
    3. EXTRACT DEFENSIVELY. Models wrap JSON in ```json fences, add a friendly
       sentence before it, and occasionally do both.
    4. VALIDATE EVERY FIELD. Present, right type, allowed value. A model that
       returns "yes" where you wanted `true` has not returned your schema.
    5. RETRY ONCE, with the parse error appended. The model usually fixes its
       own mistake when you show it the mistake. Once — not in a loop. A loop
       against a 50-requests-per-day quota is a bad afternoon.

Watch the output: the first attempt comes back fenced and with an invalid
urgency value, validation catches it, and the retry fixes it.
"""

import argparse
import json
import re
from typing import Any

from ejo.client import get_client, describe_client

# Step 2: the allowed values, defined once, in code, so validation and the
# prompt can never drift apart.
URGENCY = ["low", "medium", "high"]
DEPARTMENTS = ["general", "maternity", "pharmacy", "emergency", "counselling"]

PROMPT = """\
You are a triage assistant for a health clinic in Rwanda.
Read the patient message and return JSON only. No prose, no code fences.

Return exactly this shape:
{{"urgency": "low", "department": "general", "summary": "...", "callback_required": false}}

Rules:
- "urgency" must be one of: {urgency}
- "department" must be one of: {departments}
- "summary" is one sentence, at most 20 words
- "callback_required" is a JSON boolean, true or false

PATIENT MESSAGE: {message}"""

MESSAGE = ("Good morning. My child has had a high fever since yesterday "
           "and has been crying a lot.")


# --- Step 3: defensive extraction -----------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    """Get a JSON object out of a reply that may not be pure JSON.

    Two things go wrong in practice and both are handled here: the reply is
    wrapped in a markdown code fence, or there is a sentence of chat before
    or after the object. We strip fences, then take the outermost balanced
    braces. If that still is not JSON, we raise — and the caller retries.
    """
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in the reply")

    return json.loads(text[start:end + 1])


# --- Step 4: validate every field -----------------------------------------

def validate(data: dict[str, Any]) -> list[str]:
    """Return a list of problems. Empty list means the data is usable.

    Returning the problems rather than raising on the first one matters: we
    want to hand the model *all* its mistakes in the retry, not make it guess
    them one request at a time.
    """
    problems: list[str] = []

    if data.get("urgency") not in URGENCY:
        problems.append(f'"urgency" was {data.get("urgency")!r}; must be one of {URGENCY}')

    if data.get("department") not in DEPARTMENTS:
        problems.append(
            f'"department" was {data.get("department")!r}; must be one of {DEPARTMENTS}'
        )

    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append('"summary" must be a non-empty string')
    elif len(summary.split()) > 20:
        problems.append(f'"summary" was {len(summary.split())} words; the limit is 20')

    if not isinstance(data.get("callback_required"), bool):
        problems.append(
            f'"callback_required" was {data.get("callback_required")!r}; '
            f"must be a JSON boolean (true/false), not a string"
        )

    return problems


# --- Step 5: one retry, carrying the error --------------------------------

def triage(client: Any, message: str, *, full: bool = False) -> dict[str, Any]:
    prompt = PROMPT.format(
        urgency=URGENCY, departments=DEPARTMENTS, message=message
    )

    for attempt in (1, 2):
        reply = client.ask(prompt)
        if full:
            print(f"--- attempt {attempt}: raw reply " + "-" * 34)
            print(reply.text)
            print()

        try:
            data = extract_json(reply.text)
            problems = validate(data)
        except (ValueError, json.JSONDecodeError) as exc:
            data, problems = {}, [f"the reply was not valid JSON: {exc}"]

        if not problems:
            if full:
                print(f"valid on attempt {attempt}")
            return data

        if full:
            print("rejected:")
            for problem in problems:
                print(f"  - {problem}")
            print()

        if attempt == 2:
            # Two failures is a real failure. Fall back to something safe —
            # here, route to a human. Session 3: handle AI failure gracefully.
            raise ValueError("model did not return valid JSON twice; route to a human")

        # The retry prompt is the original prompt plus what went wrong. Note
        # that OfflineMockClient.RETRY_MARKER matches this first line.
        prompt = (
            prompt
            + "\n\nYour previous reply could not be used. Fix these problems and "
            + "return JSON only:\n"
            + "\n".join(f"- {p}" for p in problems)
        )

    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Return a validated JSON response for a clinic triage message.",
    )
    parser.add_argument("--full", action="store_true", help="print raw attempts and validation errors")
    args = parser.parse_args()

    client = get_client()

    if args.full:
        print(describe_client(client))
        print()

    result = triage(client, MESSAGE, full=args.full)

    print("STEP 3 · Structured output")
    print("-" * 68)
    print(f"Patient message: {MESSAGE}")
    print()
    print("Validated JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    if result["urgency"] == "high":
        print(f"Route: {result['department']} | callback: {result['callback_required']}")


if __name__ == "__main__":
    main()
