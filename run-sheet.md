# Session 5 — AI Integration & RAG
## Instructor run sheet · 60 minutes · Tuesday 11 August 2026

**The story of the hour, in four beats:**

1. The model makes up a phone number. (minute 1)
2. We paste the document into the prompt. It gets it right. (minute 4)
3. So why isn't that the end? Three reasons — it stops fitting, it costs
   money, and it gets *worse*. (minute 6)
4. Everything after that is one idea: **send only the part of the document
   that answers the question.**

**Everything is in English** — all demos, documents and questions.

**The sentence they should leave with:**
> RAG = find the relevant pages, paste them into the prompt, then ask.

**The terminal is the main surface.** Slides carry only what you cannot run.

---

## Before you start

### The night before

```bash
cd code
python3 setup_check.py
python3 01_ask_without_rag.py && python3 08_rag.py && python3 11_evaluate.py
```

Check against the appendix at the bottom of this sheet. If numbers differ,
something in `docs/` changed.

Send students the `code/` folder tonight, with one instruction: **run
`python3 setup_check.py` before bed.**

### Ten minutes before

- Terminal font **20pt or larger**. Check it from the back row.
- Window about **74 columns** — every script is wrapped to fit.
- `export EJO_OFFLINE=1`. Do this even if your key works. Deterministic beats
  live, and the timings below assume it.
- Two terminal tabs: one in `code/`, one spare.
- Clean deck on slide 1. **Presenter notes off** — students photograph slides.
- `clear` between every script.

### If everything breaks

Every script already runs offline with no key. If the *laptop* dies, the clean
deck carries the 14-slide story and every number is in the appendix. Print it.

---

## Minute by minute

### 0–3 · Cold open. Terminal only. No title slide.

Do not introduce yourself yet.

```bash
python3 01_ask_without_rag.py
```

Let it print. **Pause three seconds** while they read the number.

> "This is a real clinic in Kanombe. I asked which number to call at the
> weekend. It gave me one. It is confident, it is polite, it is correctly
> formatted — and it is invented. The real number is 0788 123 456."

> "Look at *how* it failed. It did not say 'I don't know'. Nothing in that
> prompt gave it permission to say 'I don't know', so it filled the silence."

**Ask the room:** *Who would have believed that number?*

→ **clean slide 1** (title), ten seconds. → **clean slide 3** (no context vs context).

### 3–6 · The simple fix, which already works

```bash
python3 02_add_context.py
```

> "Same question. One change: I pasted the clinic handbook into the prompt
> before asking. That is it. That is the whole idea, and it already works."

Walk the three sections quickly — **A** plain text, **B** JSON, **C** read
from a file — and make the point that they are the same thing:

> "'Attaching a file' means reading the file and putting the text in the
> prompt. Nothing is uploaded. There is no file. It is a big string."

JSON is worth ten seconds: use it when you send several documents and want the
model to be able to say *which one* it used.

→ **slide 3**.

### 6–9 · So why is that not the end?

Stay on the last section of `02`. It is already on screen.

```
the question        37 characters
the document     1,311 characters
what we sent     1,480 characters
```

> "I sent the whole handbook to answer one question about one line of it. And
> I will send it again for the next question. You pay for every character,
> every time."

→ **slide 4**. Three things break as documents grow:

1. **It stops fitting.** Context windows are finite — Session 4. A 40-page
   handbook does not fit. Two hundred documents certainly do not.
2. **It gets expensive.** Billed per token, per request. Free tier is 50
   requests a day.
3. **It gets worse.** Bury one useful line in forty pages and the model is
   more likely to miss it. More context is not more accuracy.

> "So the plan for the rest of the hour is one sentence: **send only the part
> of the document that answers the question.**"

### 9–13 · Take the word apart

→ **slide 5**. Three separate things, and only one is AI:

- **Retrieval** — search your own text, pull back the closest pieces.
  **No AI generation happens here.**
- **Augmented** — those pieces go into the prompt beside the question.
- **Generation** — the model reads that prompt and writes the answer.

→ **slide 6**, and leave it up:
> **RAG = find the relevant pages, paste them into the prompt, then ask.**

> "Not a model. Not a library. Not a product. It is an architecture — a
> pattern for what you put in the prompt."

### 13–16 · "Why not just…?"

→ **slide 7** (the comparison table). Spend most of it on fine-tuning:

> **Fine-tuning changes *how* the model speaks. RAG changes *what* it knows.**

### 16–25 · The heart of it: how prompting changes

→ **slides 8, 9, 10** — nine minutes. Do not rush this.

**Slide 8 — the prompt you wrote before.** Prose. Role, tone, instruction.
You are hoping the knowledge is in there.

**Slide 9 — the prompt you write now.** Read it out line by line. Four things
changed:

1. **A context slot.** Assembled at runtime, not written by hand. You are no
   longer prompting an assistant; you are prompting **a reader**.
2. **Instructions about the context, not about the world.** *Use only the
   context below. Cite the item you used.*
3. **Explicit permission to fail.** `If the answer is not in the context, say
   you don't know.`
   > "That one sentence is the highest-leverage thing in the whole prompt.
   > Without it the model fills the silence, because nothing told it not to."
4. **Design for retrieval failure**, not just model failure. When the pieces
   are wrong, say so and show the sources.

**Slide 10 — the anatomy.** Have them write it down:

```
1. INSTRUCTION   what to do, and what to do when you can't
2. CONTEXT       the retrieved pieces, numbered, with their source
3. QUESTION      the user's actual words, unmodified
4. OUTPUT RULE   language, length, format, citation
```

Three details, thirty seconds each:
- **Number the pieces and label the source** — lets the model cite, lets you
  find which piece caused a wrong answer.
- **Question last.** Instructions top, context middle, question bottom.
- **Never reword the user's question** — you throw away the words retrieval
  matched on.

> "The prompt is now code. Template, variables, a test. Version it."

### 25–29 · The API, and getting JSON back

→ **clean slide 4** first, 45 seconds.

> "It is an HTTP POST with a JSON body. Same `requests` as Session 2, same
> status codes you saw in Postman, key in a header."

401 key · 429 quota · 5xx retry with backoff. **Never retry 401 or 422.**

Two things worth saying out loud:
- `messages` is a **list** because the server is stateless. The model
  remembers nothing, so you resend the conversation every turn — which is why
  long chats get expensive.
- The key lives in **`.env`. Never in code, never in git, never in front-end
  JavaScript.** For 25 students on 50 requests/day: one key on one small
  server that everyone's app calls.

Then terminal:

```bash
python3 03_structured_output.py
```

> "An `if` statement cannot read a paragraph."

Attempt 1 comes back wrapped in a code fence, with `"urgency": "VERY_HIGH"` —
never an allowed value — and `"callback_required": "yes"`, a string where you
wanted a boolean. Validation catches both, the retry carries the errors back,
attempt 2 is clean.

→ **clean slide 5**. **Extract defensively. Validate every field. Retry once.**

### 29–32 · Streaming

**Ask first: which one finishes first?**

```bash
python3 04_streaming.py
```

Neither. Both 3.33 s. What changed is **time to first token**: 3.33 s versus
0.12 s.

> "Streaming does not make it faster. It makes it *feel* finished sooner.
> Three seconds of blank screen reads as broken. Three seconds of arriving
> text reads as fast. That is a product decision, not a performance one."

Be honest, on the record:
> "EjoChat's public docs describe no streaming endpoint, so I am not going to
> show you one. This is the pattern, and the FastAPI code is real."

→ **clean slide 6**.

### 32–36 · Chunking

```bash
python3 05_chunking.py
```

`load → clean → split → embed → store → retrieve`

Point at the seam. This is the money shot:

```
chunk [0] ends   …'ekend, at night, or on a public holiday, c'
chunk [1] starts …'all the emergency number: 0788 123 456. Th'
```

> "The word *call* is cut in half. The fact exists nowhere in one piece. No
> model and no clever query can find it. Retrieval is working perfectly and
> the answer is unreachable."

Then overlap on, and it comes back. That is the entire reason overlap exists.

Size trade-off in one breath: too small is a fragment with no context, too big
buries one useful line among nine. Start at 500 characters and tune **by
reading the output**.

### 36–42 · Embeddings — six minutes, and worth them

→ **clean slide 8** (the ladder), then terminal.

1 number is a point on a line. 2 is a point on a page. 3 is a point in this
room. 384 is a space nobody can draw — and *which two are closest* is the same
arithmetic.

```bash
python3 06_embeddings.py
```

Four sections. Walk them in order:

1. **One embedding.** Eight numbers on screen. *"That is the whole object. No
   words are kept inside it. You cannot turn it back into the sentence."*
2. **The grid.** Every sentence against every other. Check the diagonal first —
   everything scores 1.00 against itself.
3. **The map.** Three facts, each written twice in different words. The pairs
   sit together. *"It looks like it understands meaning."*
4. **Take the digits out.** The gap collapses by **66%**.

> "Deleting the digits removed nothing from the meaning — the sentences still
> say the same things. What it removed was the same phone number appearing in
> both halves of a pair. So most of what looked like understanding was
> character matching."

**The sentence to land:**
> "When something looks like it works, find out *why* before you believe it."

Stay on **clean slide 8**. Then, to involve the room:

```bash
python3 06_embeddings.py "a sentence someone gives you" "the same thing reworded"
python3 06_embeddings.py "that same sentence" "something unrelated"
```

Ask for a Kinyarwanda pair from the room and run it. **You do not need to
speak the language** — they supply the sentences, you run the tool and read
the two numbers out. It makes the point better than any claim you could make,
and it is honest: nobody in that room has measured this either.

### 42–45 · Retrieval on its own

**Ask before you press enter:** *three documents, about twenty pieces. Which
one wins, and by how much?*

```bash
python3 07_retrieve.py
```

Top hit 0.360, second 0.225. **No AI has run.** This is sorting.

Three rules of retrieval debugging:
- **The gap matters more than the score.** 0.61 vs 0.60 means it is guessing.
- **An absolute score means nothing alone.** Never hard-code a threshold you
  have not measured.
- **This is a linear scan** — every piece, every query. Remember that at 50.

### 45–50 · Put it together

```bash
python3 08_rag.py
```

Walk A → E. **Section C is why this script exists: the assembled prompt, in
full, on the projector.** Sit on it.

> "This string is what actually gets sent. Everything mysterious about RAG
> stops being mysterious once you read the string."

Then section D, and land the callback:

> "Same model. Same question I asked you at minute one, when it invented
> 0788 300 200. The only thing that changed is what went into the prompt —
> and unlike minute four, we sent one paragraph instead of the whole book."

Section E — sources.
> "Always show these. One line of code, and it is how your user catches you
> being wrong."

### 50–53 · Vector databases

> "Step 7 compared the question to every piece, one at a time. Instant at 20.
> Fine at 20,000. Hopeless at 2 million. That is the whole reason vector
> databases exist: they keep an index so lookup is sub-linear."

→ **clean slide 9**, then:

```bash
python3 10_vector_db_pgvector.py
```

> "You already run PostgreSQL. `CREATE EXTENSION vector` and it is a vector
> database. One less system to deploy, and your pieces can join against your
> ordinary tables."

Two traps, sixty seconds:
- **The distance metric.** `<=>` is cosine, `<->` is L2, and the default is
  often L2 — plausible, wrongly-ordered results.
- **The default embedding function is usually English-only.** *The database
  does not understand your language. The embedding model does.*

Name Chroma, Qdrant, Weaviate, Pinecone once and move on.

### 53–56 · When it still gets it wrong

```bash
python3 08_rag.py "How much does an X-ray cost?"
```
Refuses. Nothing matched well enough — the *say you don't know* line earning
its place.

```bash
python3 08_rag.py "When did the clinic first open?"
```

> "Not in any document. But the word *open* is all over the opening-hours
> section, so retrieval hands that back, and the answer comes out confident,
> grounded, cited — and wrong. No prompt wording fixes this. The bug is in
> retrieval, not in generation."

→ **clean slide 12** — the four ways:
1. Retrieval returned the wrong piece. Grounded *and* wrong is more convincing
   than a plain guess, so it is more dangerous.
2. The answer is not in the documents at all.
3. The documents are stale. RAG serves last year's fees with total confidence.
4. Retrieval worked and the model ignored it.

**And when not to use RAG:** if the answer is one row in a table, run the
query. RAG is for prose, not for `SELECT * FROM users WHERE id = 7`.

### 56–58 · Measure it

```bash
python3 11_evaluate.py
```

```
random (null model)   R@1 0.00   R@3 0.12   MRR 0.12
hashing baseline      R@1 0.56   R@3 0.94   MRR 0.75
```

> "`random` is deterministic nonsense. It is there so you know what zero
> knowledge looks like. A 500 MB multilingual model that ties with random does
> not understand your documents, and nothing downstream will fix that."

Sixteen questions, each labelled with the document that answers it. An
afternoon of work. **That is the difference between engineering and hoping.**

### 58–60 · Homework

→ **clean slide 13**

- point the pipeline at **3–5 documents from your own capstone**
- **10–15 labelled questions, written before you measure**
- submit **one question it answers correctly and one it answers wrongly**
- **two sentences on why the wrong one was wrong** — that is 35% of the mark

> "I am not marking whether it works. I am marking whether you can explain why
> it didn't."

Next: Friday is problem framing. Session 7 is agents — this, plus the model
deciding *when* to retrieve.

---

## If you are running late

| Cut | Saves | Instead |
|---|---|---|
| 1. `10` live run | 3 min | Say the pgvector sentence over **clean slide 9** |
| 2. `04` live run | 3 min | **Clean slide 6** has the streaming explanation |
| 3. `03` live run | 3 min | **Clean slide 5** has the JSON shape |
| 4. `06` sections 1–2 | 3 min | Go straight to the map and the digits table |
| 5. `05` live run | 4 min | Read the seam out of the appendix |

**Never cut:** the cold open, `02` (the simple version that works), minutes
16–25, section C of `08`, or the two failure questions.

Checkpoints — **minute 25**: prompt anatomy done. **Minute 50**: `08` done.
If you are eight minutes over at minute 45, go straight from `08` to the
failure questions to homework.

## If you are running early

- Ask the room to write the four-part RAG flow from memory, then show clean slide 10.
- Take a question from the room and run `python3 07_retrieve.py "their question"`.
- Ask for a sentence pair in any language and run `06_embeddings.py` on it.
- Open `ejo/store.py` and read `search()` aloud. Fifteen lines. Seeing that
  "vector search" is a sort is worth more than another concept.

---

## Appendix — every number, for narrating without a laptop

```
01  invents            0788 300 200
02  correct answer     0788 123 456   (all three shapes: text, JSON, file)
    the cost           question 37 chars · document 1,311 · sent 1,480

04  blocking   time to first token 3.33s   total 3.33s
    streamed   time to first token 0.12s   total 3.33s

05  chunk_size=295, overlap=0  -> the fact does not survive
    chunk [0] ends   'ekend, at night, or on a public holiday, c'
    chunk [1] starts 'all the emergency number: 0788 123 456. Th'
    chunk_size=295, overlap=50 -> it survives

06  same meaning 0.275 · different fact 0.138 · the gap 0.138
    delete the digits and the gap drops to 0.046   ->   -66%
    per pair: phone -56%   fees -28%   jabs -29%

07  0.360 Emergency contact | 0.225 second | 0.191 third

08  "If you need help at the weekend, at night, or on a public holiday,
     call the emergency number: 0788 123 456."
    source [1] kanombe-clinic.md · Emergency contact (0.360)

11  random   R@1 0.00  R@3 0.12  R@5 0.44  MRR 0.12
    hashing  R@1 0.56  R@3 0.94  R@5 0.94  MRR 0.75
    7 of 16 not ranked first — read those out, they are the lesson
```

Verified on a clean run, 10 August 2026, `EJO_OFFLINE=1`, default `hashing`
backend, Python 3.13.
