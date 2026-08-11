# Session 5 homework — Make your capstone answer questions about its own documents

**Due:** before Session 7 (Tuesday 18 August 2026)
**Time:** about three hours, most of it reading output rather than writing code

---

## What you are building

The published lab is *"turn your capstone project into an AI-powered assistant
in Kinyarwanda using the EjoChat API."* Concretely:

Point the pipeline from session 5 at **3–5 real documents from your own
capstone domain**, and get it answering questions about them.

Not the clinic. Yours. If your capstone is about student housing, use the
housing rules. If it is about farming, use the extension service leaflets. The
documents must be ones you did not write for this exercise — real text, with
real inconsistencies in it.

---

## Steps

**1. Collect your documents.** 3–5 files, plain text or markdown. Convert PDFs
however you like. Put them in `docs/`.

**2. Point the pipeline at them.** In `07_retrieve.py`, `08_rag.py` and
`11_evaluate.py`, change the `DOCS` list. That is the only code change
required to get something working.

```bash
python3 08_rag.py "a real question about your documents"
```

**3. Build an evaluation set.** 10–15 questions in `eval/questions.jsonl`,
each labelled with the document and section that answers it. Write the
questions **first**, before you look at how well retrieval does — otherwise
you will unconsciously write questions your pipeline already handles.

Include at least one question your documents **cannot** answer.

If your capstone is meant to work in Kinyarwanda, write some of your questions
in Kinyarwanda too — and read `LOCALISATION.md` first. The embedding model, not
the database and not the LLM, is what decides whether your language works.

**4. Measure.**

```bash
python3 11_evaluate.py random hashing
```

If you managed the download, add a real model:

```bash
python3 11_evaluate.py random hashing st:paraphrase-multilingual-MiniLM-L12-v2
```

---

## What you submit

A single markdown file. Four short sections.

### 1. Your setup (3 lines)
What the documents are, how many chunks, which embedding backend.

### 2. One question it answers CORRECTLY
Paste the question, the answer, and the retrieved sources with their scores.

### 3. One question it answers WRONGLY
Same: question, answer, sources, scores.

The wrong one must be a *plausible* question — something a real user of your
project would ask. "asdfgh" does not count.

### 4. Two sentences on why it was wrong

**This is the actual assignment.** Everything above it is setup.

Not "the model hallucinated". Say which stage failed and how you know:

- Was the answer **not in your documents at all**? Then the failure is that it
  did not refuse — look at your prompt's permission-to-fail line.
- Was the answer in your documents but **the wrong chunk came back**? Then the
  failure is retrieval. Print the top 5 with scores. Was the gap between #1
  and #2 small — was it guessing?
- Was the answer **cut across a chunk boundary**? Print the chunks and check.
- Was the right chunk retrieved and the model **ignored it**? Then it is the
  prompt.

Those are four different bugs with four different fixes, and telling them
apart is the skill this session was about.

---

## Marking

| | |
|---|---|
| Pipeline runs on your own documents | 20% |
| Evaluation set: 10–15 questions, properly labelled, written before measuring | 25% |
| The two examples, with sources and scores shown | 20% |
| **The explanation of the wrong answer** | **35%** |

You are not marked on whether retrieval works well. A project with recall@3 of
0.4 and a sharp diagnosis scores higher than one with 0.9 and "it works".

---

## Rules

- **Do not commit your API key.** `.env` is git-ignored; check it stayed that
  way before you push. A key in a public repository is a key on somebody
  else's daily quota.
- **You do not need an approved key.** Everything runs offline with the mock.
  If you have a key, remember the free tier is 50 requests per day — the
  offline mock exists so you do not spend them debugging a chunker.
- **You do not need to download an embedding model.** The `hashing` backend
  needs no download and is a legitimate baseline. If you did download one,
  compare them and say which won on your corpus.

---

## If you get stuck

Work through the pipeline in order, because the stages fail differently:

```
load  →  clean  →  split  →  embed  →  store  →  retrieve  →  prompt  →  generate
```

```bash
python3 setup_check.py     # is the environment fine?
python3 05_chunking.py     # are the chunks sensible? print them and read them
python3 07_retrieve.py "…" # does the right chunk come back, and with what gap?
python3 08_rag.py "…"      # is the assembled prompt what you think it is?
```

Nine times out of ten the answer is in `05` or `07`, and you find it by
reading the output rather than by changing the code.
