# Running this in Kinyarwanda

Everything in this repository is in English, because that is the language the
session is delivered in. The technique does not care what language you use.

## Why the language question matters anyway

An embedding model is a separate model from the LLM, and it decides whether
retrieval finds the right piece of your document. Most embedding models were
trained mostly on English.

Two facts you can check:

- `paraphrase-multilingual-MiniLM-L12-v2` publishes a list of 50 languages.
  Kinyarwanda is not on it.
- LaBSE publishes a list of 109 languages. Kinyarwanda is on it.

Both are claims about training data. Neither is a measurement of how well the
model retrieves *your* documents. **When a model does not know a language,
nothing crashes.** No error, no warning — you quietly get the wrong chunks and
a confident answer written on top of them.

## Test it yourself, in a language you speak

```bash
python3 06_embeddings.py "a sentence" "the same thing, reworded"
python3 06_embeddings.py "a sentence" "something completely unrelated"
```

You need both numbers. A single similarity score tells you nothing — you are
looking for a clear gap between the two.

This works in any language. If you speak Kinyarwanda, run it with a
Kinyarwanda pair and you will learn more in thirty seconds than any blog post
will tell you.

## Switching the whole pipeline over

1. Replace the files in `docs/` with your own documents, in your own language.
2. Rewrite `eval/questions.jsonl` — same format, questions in your language,
   each labelled with the document and section that answers it.
3. Run the measurement:

```bash
python3 11_evaluate.py random hashing
```

`random` is the null model. If a downloaded multilingual model cannot beat it
on your corpus, that model does not understand your language, and nothing you
do downstream will repair that.

## A note on the default backend

`EJO_EMBEDDER=hashing` compares characters, not meaning. That makes it a
reasonable baseline for a language with rich prefixes and suffixes, because
character n-grams handle word-form changes better than whole-word matching.
It is still not a meaning model. Step 6 shows you exactly how that limitation
shows up.
