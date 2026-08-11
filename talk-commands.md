# Talk Commands

Run these from the repository root unless the command says otherwise.

## Start

```bash
cd code
export EJO_OFFLINE=1
python3 setup_check.py
```

## Clean Flow

```bash
python3 run_lesson.py quick
```

This runs the main demos in a short teaching order:

1. ask without RAG;
2. add document context;
3. RAG answer with sources;
4. embeddings;
5. structured output;
6. streaming.

## One Lesson At A Time

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

Names also work:

```bash
python3 run_lesson.py rag
python3 run_lesson.py embeddings
python3 run_lesson.py streaming
python3 run_lesson.py structured
```

## Rehearsal

```bash
python3 run_lesson.py quick --no-pause
python3 run_lesson.py all --no-pause
```

## Slides

From the `code/` folder on macOS:

```bash
open ../slides/session-5-rag-clean.pptx
```
