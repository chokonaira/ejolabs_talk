# Ejo Labs Session 5: AI Integration, Structured Output, and RAG

This repo is for a beginner-friendly workshop on how AI apps connect to an
LLM, return useful data, stream responses, and use documents with RAG.

The examples use one simple scenario: a student asks questions about a small
clinic in Kanombe. At first the assistant guesses. Then we give it trusted
notes, retrieve the right note, and make the answer more grounded.

Everything runs offline by default. You do not need an API key, internet, or
extra packages to start.

## Quick Start

Clone the repo and enter the code folder:

```bash
git clone https://github.com/chokonaira/ejolabs_talk.git
cd ejolabs_talk/code
```

Check that Python can run the demos:

```bash
python3 setup_check.py
```

Run the short workshop flow:

```bash
python3 run_lesson.py quick
```

If `python3` works, you are ready.
The rest of the commands below assume your terminal is still inside `code/`.

If your terminal says `python: command not found`, use `python3`. If it says
`python3: command not found`, install Python 3.10 or newer from
<https://www.python.org/downloads/>.

## Run One Lesson

Use this when you want to understand one topic at a time:

```bash
python3 run_lesson.py 1   # ask without RAG
python3 run_lesson.py 2   # add document context
python3 run_lesson.py 3   # structured output / JSON
python3 run_lesson.py 4   # streaming
python3 run_lesson.py 5   # chunking
python3 run_lesson.py 6   # embeddings
python3 run_lesson.py 7   # retrieval
python3 run_lesson.py 8   # RAG answer with sources
python3 run_lesson.py 9   # evaluate retrieval
```

You can also use names:

```bash
python3 run_lesson.py rag
python3 run_lesson.py embeddings
python3 run_lesson.py streaming
python3 run_lesson.py structured
```

To see the full menu:

```bash
python3 run_lesson.py list
```

## What Each Lesson Teaches

| Lesson | Topic | What you should notice |
|---|---|---|
| 1 | Ask without RAG | The assistant sounds confident but invents a phone number. |
| 2 | Add context | When the trusted note is placed in the prompt, the answer becomes correct. |
| 3 | Structured output | JSON gives your app fields like `answer`, `confidence`, and `sources`. |
| 4 | Streaming | The full answer does not finish faster, but the user sees progress sooner. |
| 5 | Chunking | Long documents must be split carefully, or the answer can be separated from the context. |
| 6 | Embeddings | Text is turned into numbers so similar meanings can be compared. |
| 7 | Retrieval | Search the document chunks before asking the LLM to write. |
| 8 | RAG | Retrieve useful chunks, add them to the prompt, then generate the answer. |
| 9 | Evaluation | Measure if retrieval is actually finding the right chunk. |

## How The Document Example Works

The sample documents are plain Markdown files in:

```text
code/docs/
```

The RAG scripts do not magically know those notes. They do this:

1. read the files from `code/docs/`;
2. split the text into smaller chunks;
3. turn each chunk into an embedding;
4. compare the student's question with the chunks;
5. put the best chunks into the prompt;
6. ask the assistant to answer only from those chunks.

That is the core idea of RAG: the model does not need to memorize your files.
Your app finds the useful part of the file and sends that part with the
question.

Try the main RAG demo:

```bash
python3 run_lesson.py 8
```

Then ask a different question:

```bash
python3 08_rag.py "How much is a malaria test?"
python3 08_rag.py "When did the clinic first open?"
```

The second question is useful because the documents do not contain the real
answer. It shows why RAG still needs testing.

## What An LLM API Call Looks Like

An API is just a request your code sends to another service.

This is the shape of a simple LLM request:

```js
await fetch("https://api.ejolabs.com/api/v1/subiza", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "kgpt_demo_key_for_class_only"
  },
  body: JSON.stringify({
    messages: [
      { role: "user", content: "Explain RAG in one sentence." }
    ]
  })
});
```

`kgpt_demo_key_for_class_only` is a fake key. Never put a real API key in a
frontend app that runs in the browser. For a real project, keep the key on a
backend server and let the frontend call your server.

The Python version used by the demos is in `code/ejo/client.py`.

## Offline Mode And Real API Keys

By default, the scripts use an offline mock so everyone can run the class
without waiting for accounts or internet.

To force offline mode:

```bash
export EJO_OFFLINE=1
```

To try a real key later:

```bash
cp .env.example .env
```

Then open `.env` and set:

```bash
EJO_API_KEY=kgpt_your_real_key_here
EJO_OFFLINE=
```

Do not commit `.env`. It is ignored by git.

## Slides And Handout

Open the clean slide deck:

```bash
open ../slides/session-5-rag-clean.pptx
```

If you are not on macOS, open this file manually:

```text
slides/session-5-rag-clean.pptx
```

The one-page student handout is here:

```text
handout/session-5-handout.pdf
```

## If You Want More Detail

Most scripts keep the output short so the lesson is easy to follow. Some have
a `--full` option when you want to inspect the prompt or debug details:

```bash
python3 02_add_context.py --full
python3 08_rag.py --full
```

The code README has a more detailed script map:

```text
code/README.md
```

## Folder Map

```text
code/                         Python demos
code/docs/                    sample documents used for RAG
slides/session-5-rag-clean.pptx
handout/session-5-handout.pdf
homework.md                   capstone homework
run-sheet.md                  instructor run sheet
talk-commands.md              quick command sheet
```
