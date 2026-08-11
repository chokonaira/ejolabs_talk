# Code Demos

From the repository root:

```bash
cd code
python3 setup_check.py
python3 run_lesson.py quick
```

If your terminal is already inside `code/`, skip `cd code`.

No API key is needed. The scripts use a local mock unless you add a real key.

## Main Commands

```bash
python3 run_lesson.py list
python3 run_lesson.py 1
python3 run_lesson.py 2
python3 run_lesson.py rag
python3 run_lesson.py quick
```

## Lesson Map

| Command | Topic | What it shows |
|---|---|---|
| `python3 run_lesson.py 1` | LLM without data | The assistant can sound right and still invent facts. |
| `python3 run_lesson.py 2` | Context | Add trusted notes to the prompt so the answer is grounded. |
| `python3 run_lesson.py 3` | Structured output | Ask for JSON so your app can read fields safely. |
| `python3 run_lesson.py 4` | Streaming | Show answer pieces as they arrive instead of waiting silently. |
| `python3 run_lesson.py 5` | Chunking | Split documents without losing the answer at the boundary. |
| `python3 run_lesson.py 6` | Embeddings | Turn text into vectors so similar text can be compared. |
| `python3 run_lesson.py 7` | Retrieval | Search document chunks before the LLM writes. |
| `python3 run_lesson.py 8` | RAG | Retrieve chunks, place them in the prompt, then answer with sources. |
| `python3 run_lesson.py 9` | Evaluation | Check if retrieval finds the correct chunk. |

## The Sample Documents

The documents are in `docs/`:

```text
docs/kanombe-clinic.md
docs/kanombe-fees.md
docs/kanombe-services.md
```

RAG works by reading those files, splitting them into chunks, searching for the
chunks closest to the question, and putting those chunks into the prompt.

Run:

```bash
python3 07_retrieve.py "What number do I call at the weekend?"
python3 08_rag.py "What number do I call at the weekend?"
```

## Real API Key Later

For class, offline mode is enough:

```bash
export EJO_OFFLINE=1
```

For a real key later:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
EJO_API_KEY=kgpt_your_real_key_here
EJO_OFFLINE=
```

Never commit `.env`.

## Optional Packages

The required demos run with Python only. Optional packages unlock extra things:

```bash
pip install -r requirements.txt
```

You only need this if you want to call a live API, try real embedding models,
or run the vector database examples.
