"""
Calling the EjoChat API — and an offline stand-in for when you cannot.

WHAT THE PUBLIC DOCS SAY (https://ejolabs.com/en/api)

    POST https://api.ejolabs.com/api/v1/subiza
    X-API-Key: kgpt_...
    {"messages": [{"role": "user", "content": "..."}]}

    Errors: 400 INVALID_REQUEST, 401 UNAUTHORIZED, 402 PAYMENT_REQUIRED,
            403 FORBIDDEN, 422 VALIDATION_ERROR, 429 RATE_LIMITED,
            500/502 INTERNAL_ERROR, 503 PROVIDER_UNAVAILABLE

    Ejo Free: 50 requests/day, 1,500 response tokens.
    Business: 3,500 requests/day, 2,048 response tokens.
    Approved accounts start with a 48-hour trial: 10 req/min, 100 req/day.

WHAT THE PUBLIC DOCS DO NOT SAY, AND THEREFORE NEITHER DO WE

    * The exact shape of the response JSON. We look for the keys that services
      like this normally use and fall back to the raw payload. Print
      `response.raw` the first time you get a real key, then tighten this.
    * A streaming mode. There is none documented. 04_streaming.py teaches the
      pattern honestly rather than inventing an endpoint.
    * An embeddings endpoint. There is none documented. Embeddings come from a
      separate model — see ejo/embed.py.

WHY THERE IS A `messages` LIST AND NOT A `message` STRING

    Because the server is stateless. The model has no memory of your last
    request. If you want it to remember the conversation, you resend the
    conversation — every turn, every time. That list is the memory, and it is
    also why long chats get expensive: you pay for the whole history again on
    every message.

WHERE THE KEY LIVES

    In `.env`, read through the environment. Never in the source. Never in git.
    Never in front-end JavaScript, where anyone can open developer tools and
    read it. For a class of 25 students sharing a 50-requests-per-day free
    tier, the right answer is one key on one small server that your apps call —
    which is the same architecture that keeps the key out of the client anyway.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

ENDPOINT = "https://api.ejolabs.com/api/v1/subiza"

# Status codes where trying again might work, and codes where it never will.
RETRYABLE = {429, 500, 502, 503}
FATAL = {400, 401, 402, 403, 422}


class EjoChatError(RuntimeError):
    """Raised for an API error, with the guidance a beginner actually needs."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"{status} — {ADVICE.get(status, 'Unexpected error')}\n{body[:400]}")


ADVICE: dict[int, str] = {
    400: "INVALID_REQUEST: the body did not match what the API expects. Print it.",
    401: "UNAUTHORIZED: missing or wrong X-API-Key. Is EJO_API_KEY set in .env?",
    402: "PAYMENT_REQUIRED: the 48-hour trial ended, or billing is inactive.",
    403: "FORBIDDEN: this key exists but is not allowed to call this API.",
    422: "VALIDATION_ERROR: a field is missing, or you exceeded a size limit.",
    429: "RATE_LIMITED: quota used up. Free tier is 50 requests/day. Back off.",
    500: "INTERNAL_ERROR: their side. Retry with backoff.",
    502: "INTERNAL_ERROR: their side. Retry with backoff.",
    503: "PROVIDER_UNAVAILABLE: upstream is down. Retry with backoff.",
}


@dataclass
class Reply:
    """What we hand back to the caller."""

    text: str
    raw: Any = None                       # the untouched response, for debugging
    source: str = "api"                   # "api" or "offline-mock"
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The real client
# ---------------------------------------------------------------------------

def _extract_text(payload: Any) -> str:
    """Pull the assistant's text out of a response whose shape is undocumented.

    We are guessing, and we say so. The moment you have a real key, print
    `reply.raw`, find the actual field, and replace this function with one line.
    Guessing defensively is how you write a client against docs that do not
    specify a response schema; pretending you know the schema is not.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("reply", "response", "answer", "message", "content", "text", "subiza"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                nested = _extract_text(value)
                if nested:
                    return nested
        # OpenAI-shaped fallback: {"choices": [{"message": {"content": ...}}]}
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            return _extract_text(choices[0])
        for key in ("data", "result", "output"):
            if key in payload:
                return _extract_text(payload[key])
    return json.dumps(payload, ensure_ascii=False)


class EjoChatClient:
    """A small, honest client. Roughly forty lines of real work."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = ENDPOINT,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("EJO_API_KEY", "")
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: list[dict[str, str]]) -> Reply:
        import requests  # imported here so the offline path needs no install

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {"messages": messages}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint, headers=headers, json=body, timeout=self.timeout
                )
            except requests.RequestException as exc:
                # Network failure, not an HTTP error. On a shared connection in
                # a lecture theatre this is the common one.
                last_error = exc
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    payload = response.text
                return Reply(text=_extract_text(payload), raw=payload, source="api")

            if response.status_code in FATAL:
                # Retrying a 401 just wastes your daily quota being wrong faster.
                raise EjoChatError(response.status_code, response.text)

            if response.status_code in RETRYABLE and attempt < self.max_retries:
                # Exponential backoff: 2s, 4s, 8s. Respect Retry-After if sent.
                wait = int(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait)
                continue

            raise EjoChatError(response.status_code, response.text)

        raise RuntimeError(f"Request failed after {self.max_retries} attempts: {last_error}")

    def ask(self, prompt: str) -> Reply:
        """One-shot convenience wrapper for a single user message."""
        return self.chat([{"role": "user", "content": prompt}])


# ---------------------------------------------------------------------------
# The offline stand-in
# ---------------------------------------------------------------------------

class OfflineMockClient:
    """A scripted stand-in so every demo runs with no key and no internet.

    READ THIS BEFORE YOU TRUST IT. This is not a language model. It is about
    sixty lines of string handling that reproduces the *shape* of two
    behaviours we need on screen:

      1. Asked a question with no context, it answers confidently and invents
         a phone number — because that is what a real model does with a
         question about a clinic it has never heard of. The invented number
         below is hard-coded by us, not generated. The lesson is real; the
         mechanism is faked.

      2. Given a CONTEXT block, it finds the line in that context most related
         to the question and answers from it — because that is what grounding
         does.

    It cannot paraphrase, it cannot reason, and it will look stupid if you ask
    it anything the script does not cover. Say so in the room. A demo that
    quietly pretends to be a model teaches students to trust demos.
    """

    #: Deliberately wrong, deliberately plausible, deliberately hard-coded.
    INVENTED_NUMBER = "0788 300 200"

    #: How well a context line must match the question before the mock will
    #: answer from it. Below this it says "I don't know".
    #:
    #: A real model has no such number — it decides in a way nobody can
    #: inspect. That is precisely why the instruction "if the answer is not in
    #: the context, say you don't know" is load-bearing: it is the only
    #: control you have over this decision. Here the control is a float you
    #: can see, which makes the behaviour teachable rather than magical.
    #:
    #: Measured on this corpus, and imperfect: a well-covered question scores
    #: 0.34–0.54, one the documents cannot answer scores 0.16–0.38. Those
    #: ranges OVERLAP, which is exactly why a threshold is not a safety
    #: feature. It is tuned to this corpus and would be meaningless on yours.
    GROUNDING_THRESHOLD = 0.25

    def __init__(self, latency: float = 0.4) -> None:
        self.latency = latency

    #: Marker that 03_structured_output.py appends when it retries. The mock
    #: uses it to decide whether it is on its first attempt or its second.
    RETRY_MARKER = "Your previous reply could not be used"

    def chat(self, messages: list[dict[str, str]]) -> Reply:
        prompt = messages[-1]["content"]
        time.sleep(self.latency)  # so the streaming demo has something to show

        if "JSON" in prompt and "triage" in prompt.lower():
            return self._structured(prompt)

        context = self._context_block(prompt)
        question = self._question(prompt)

        if not context:
            return Reply(
                text=(
                    f"The clinic's emergency number is {self.INVENTED_NUMBER}. "
                    f"You can call it at any time, including weekends and "
                    f"public holidays."
                ),
                source="offline-mock",
                meta={"grounded": False, "note": "scripted fabrication"},
            )

        score, line = self._best_line(context, question)
        if score < self.GROUNDING_THRESHOLD:
            return Reply(
                text="I don't know — that is not in the context I was given.",
                source="offline-mock",
                meta={"grounded": False, "score": round(score, 3),
                      "note": "refused: nothing in the context matched well enough"},
            )

        return Reply(
            text=line.strip(),
            source="offline-mock",
            meta={"grounded": True, "score": round(score, 3),
                  "note": "echoed the best-matching context line"},
        )

    def ask(self, prompt: str) -> Reply:
        return self.chat([{"role": "user", "content": prompt}])

    def _structured(self, prompt: str) -> Reply:
        """Scripted responses for the structured-output demo.

        The first reply is wrong in the two ways real models are actually
        wrong: it wraps the JSON in a markdown fence, and it invents an enum
        value that was never on the allowed list. The second reply — after our
        code retries with the parse error appended — is correct.

        Both are hard-coded here. A real model fails like this often but not
        every time, and a demo that only works sometimes is not a demo.
        """
        if self.RETRY_MARKER in prompt:
            payload = (
                '{"urgency": "high", "department": "emergency", '
                '"summary": "Child with high fever since yesterday, crying persistently.", '
                '"callback_required": true}'
            )
            return Reply(text=payload, source="offline-mock",
                         meta={"attempt": 2, "note": "scripted valid reply"})

        payload = (
            "```json\n"
            '{"urgency": "VERY_HIGH", "department": "emergency", '
            '"summary": "Child has a high fever.", '
            '"callback_required": "yes"}\n'
            "```"
        )
        return Reply(text=payload, source="offline-mock",
                     meta={"attempt": 1, "note": "scripted invalid reply"})

    # -- the sixty lines of string handling --------------------------------

    @staticmethod
    def _context_block(prompt: str) -> str:
        match = re.search(r"CONTEXT:\s*(.*?)(?:\n\s*QUESTION:|\Z)", prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _question(prompt: str) -> str:
        match = re.search(r"QUESTION:\s*(.*)", prompt, re.DOTALL)
        return match.group(1).strip() if match else prompt

    @staticmethod
    def _best_line(context: str, question: str) -> tuple[float, str]:
        """Return (score, line) for the context line closest to the question."""
        from .embed import HashingEmbedder, cosine

        # The context may arrive as plain text or wrapped in JSON. In JSON the
        # whole document is one long line with \n written out as two
        # characters, so unescape those first — otherwise the mock treats the
        # entire document as a single candidate and echoes all of it.
        # A real model reads either shape without help; this is a mock problem.
        context = context.replace("\\n", "\n")

        # Read ONLY context item [1] when the context is numbered. Retrieval
        # has already decided which chunk is best; a stand-in that quietly
        # re-ranks the context would disagree with the scores printed on
        # screen a moment earlier, and the demo would contradict itself.
        items = re.split(r"\n(?=\[\d+\])", context.strip())
        top = items[0] if items else context

        # Candidates must be whole SENTENCES, not raw lines. Documents are
        # hard-wrapped at about 78 characters, so a line is usually half a
        # fact — "call the emergency" with the number on the next line.
        # Unwrap first, then split on sentence ends.
        candidates: list[str] = []
        for block in re.split(r"\n\s*\n", top):
            unwrapped = " ".join(part.strip().strip('",') for part in block.split("\n"))
            unwrapped = re.sub(r"^\[\d+\]\s*\([^)]*\)\s*", "", unwrapped)
            for sentence in re.split(r"(?<=[.!?])\s+", unwrapped):
                sentence = sentence.strip()
                if len(sentence) > 15:
                    candidates.append(sentence)

        if not candidates:
            return 0.0, ""
        embedder = HashingEmbedder()
        question_vec = embedder.embed([question])[0]
        scored = [(cosine(question_vec, v), text)
                  for text, v in zip(candidates, embedder.embed(candidates))]
        return max(scored)


# ---------------------------------------------------------------------------
# Picking one
# ---------------------------------------------------------------------------

def get_client(force_offline: bool | None = None):
    """Return a real client if we can, the mock if we cannot.

    Offline when: EJO_OFFLINE=1 is set, or no EJO_API_KEY is present.
    That default is deliberate — a student with no approved key still runs
    every script in this repository, start to finish.
    """
    if force_offline is None:
        force_offline = os.environ.get("EJO_OFFLINE", "").strip() in {"1", "true", "yes"}

    if force_offline or not os.environ.get("EJO_API_KEY"):
        return OfflineMockClient()
    return EjoChatClient()


def describe_client(client: Any) -> str:
    """One line for the top of each script, so nobody is confused about which
    they are watching. On a projector this matters more than it sounds."""
    if isinstance(client, OfflineMockClient):
        return "provider: OFFLINE MOCK (no key / EJO_OFFLINE=1) — scripted, not a model"
    return f"provider: EjoChat live — POST {ENDPOINT}"
