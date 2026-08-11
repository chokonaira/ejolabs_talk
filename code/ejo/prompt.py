"""
The RAG prompt — the part most tutorials skip.

Once retrieval is involved, a prompt stops being a piece of prose you wrote
and becomes a TEMPLATE WITH A SLOT that your program fills in at runtime.
Four things change about how you write it:

  1. YOU ADD A CONTEXT SLOT. The prompt is assembled per request. You are no
     longer prompting a knowledgeable assistant; you are prompting a reader.

  2. YOUR INSTRUCTIONS BECOME INSTRUCTIONS ABOUT THE CONTEXT, not about the
     world. "Use only the context below." "Quote the line you used." "Say
     which section it came from."

  3. YOU GIVE IT EXPLICIT PERMISSION TO FAIL. `If the answer is not in the
     context, say you don't know.` Without that sentence the model fills the
     silence, because nothing gave it permission not to. This single line is
     the highest-leverage thing in the entire prompt.

  4. YOU DESIGN FOR RETRIEVAL FAILURE, not just model failure. When the
     retrieved chunks are irrelevant the right behaviour is to say so and show
     the sources, so the person reading can see the mistake for themselves.

THE ANATOMY — four parts, in this order:

    1. INSTRUCTION   what to do, and what to do when you can't
    2. CONTEXT       the retrieved chunks, numbered, each with its source
    3. QUESTION      the user's actual words, unmodified
    4. OUTPUT RULE   language, length, format, citation requirement

Three details inside that structure are worth the seconds they cost:

  * NUMBER THE CHUNKS AND LABEL THE SOURCE. It lets the model cite, and it
    lets you work out which chunk produced a wrong answer.
  * PUT THE QUESTION LAST. Instructions top, context middle, question bottom
    is the widely used ordering, and the question is what you want freshest.
  * DO NOT PARAPHRASE THE USER'S QUESTION. Retrieval matched on their words.
    Rewrite them and you throw away the reason those chunks came back.

And the thing to take away: THE PROMPT IS NOW CODE. It has a template, it has
variables, it has a test. Version it, review it, and diff it when the answers
get worse.
"""

from __future__ import annotations

from .store import Hit

# The template lives in one place so that changing it changes every script,
# and so that "we changed the prompt" shows up as one line in a git diff.
RAG_TEMPLATE = """\
Use ONLY the context below to answer. If the answer is not in the
context, say you don't know. Do not invent anything.
Answer in {language}, in one or two sentences, and end with the
number of the context item you used, like [1].

CONTEXT:
{context}

QUESTION: {question}"""


# The same task written for a model answering from memory. Keep it next to the
# real one: showing them side by side is the lesson.
NO_RAG_TEMPLATE = """\
You are a helpful assistant for a community health clinic in Rwanda.
Answer the user's question clearly and politely, in {language}.

Question: {question}"""


def format_context(hits: list[Hit], max_chars: int = 1200) -> str:
    """Turn retrieved chunks into the numbered CONTEXT block.

    `max_chars` is a budget, not a suggestion. Context is tokens, tokens are
    money and latency, and past a point extra context makes answers worse
    rather than better. Cut at a chunk boundary, never mid-chunk.
    """
    lines: list[str] = []
    used = 0
    for position, hit in enumerate(hits, start=1):
        entry = f"[{position}] ({hit.chunk.label()} · {hit.chunk.section}) {hit.chunk.text}"
        if used + len(entry) > max_chars and lines:
            break
        lines.append(entry)
        used += len(entry)
    return "\n".join(lines) if lines else "(nothing was retrieved)"


def build_rag_prompt(question: str, hits: list[Hit], language: str = "English") -> str:
    """Assemble the four-part prompt. This is the whole of 'augmentation'."""
    return RAG_TEMPLATE.format(
        language=language,
        context=format_context(hits),
        question=question,          # verbatim. Do not touch the user's words.
    )


def build_plain_prompt(question: str, language: str = "English") -> str:
    """The same question, asked of a model that has to answer from memory."""
    return NO_RAG_TEMPLATE.format(language=language, question=question)
