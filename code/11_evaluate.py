"""
STEP 11 — Measuring retrieval, instead of having an opinion about it.

    python3 11_evaluate.py
    python3 11_evaluate.py hashing random st:LaBSE

THIS IS THE MOST IMPORTANT SCRIPT IN THE REPOSITORY.

Everything else shows you how RAG works. This one tells you whether YOURS
works, on YOUR documents, in YOUR language. Without it you are guessing, and
in Kinyarwanda you are guessing about something nobody has measured for you.

WHAT IT MEASURES

    recall@k   Of the questions we asked, in what fraction did the correct
               chunk appear anywhere in the top k results?

               This is the right metric for RAG because it answers the only
               question generation cares about: was the answer IN the prompt?
               If the right chunk never got retrieved, no prompt wording and
               no bigger model will save you.

    MRR        Mean reciprocal rank. If the right chunk was 1st you score 1,
               2nd scores 1/2, 3rd scores 1/3. It rewards putting the right
               chunk at the top, not merely somewhere in the list.

HOW TO BUILD YOUR OWN SET — THIS IS THE HOMEWORK

    Ten to fifteen real questions, written by someone who would actually ask
    them, each labelled with the document and section that answers it. That is
    it. It takes an afternoon and it is the difference between engineering and
    hoping. `eval/questions.jsonl` is the example; yours should be about your
    own capstone documents.

WHY `random` IS IN THE DEFAULT LIST

    It is the null model — deterministic nonsense. It tells you what a score
    of "no information at all" looks like on this corpus, which is the only
    way to know whether a real model is doing anything. A multilingual model
    that ties with random on your corpus does not know your language, and no
    amount of prompt engineering downstream will repair that.
"""

import argparse
import json

from ejo.embed import get_embedder
from ejo.store import build_store

DOCS = [
    "docs/kanombe-clinic.md",
    "docs/kanombe-fees.md",
    "docs/kanombe-services.md",
]
QUESTIONS = "eval/questions.jsonl"
KS = (1, 3, 5)

DEFAULT_BACKENDS = ["random", "hashing"]


def load_questions() -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_correct(chunk, question: dict) -> bool:
    """Did we retrieve the chunk that actually answers this question?

    The label is (source document, section). We match the section loosely
    because the headings in this corpus are bilingual — "Guhamagara mu
    byihutirwa / Emergency contact" should match the label "Emergency contact".
    """
    return (
        chunk.source == question["source"]
        and question["section"].lower() in chunk.section.lower()
    )


def evaluate(spec: str, questions: list[dict]) -> dict:
    embedder = get_embedder(spec)
    store = build_store(DOCS, embedder, chunk_size=500, overlap=50)

    max_k = max(KS)
    recall = {k: 0 for k in KS}
    reciprocal_ranks: list[float] = []
    failures: list[tuple[dict, list]] = []

    for question in questions:
        hits = store.search(question["question"], k=max_k)
        ranks = [i for i, hit in enumerate(hits, start=1)
                 if is_correct(hit.chunk, question)]

        for k in KS:
            if any(rank <= k for rank in ranks):
                recall[k] += 1

        reciprocal_ranks.append(1.0 / ranks[0] if ranks else 0.0)
        if not ranks or ranks[0] > 1:
            failures.append((question, hits))

    total = len(questions)
    return {
        "backend": embedder.name,
        "dim": embedder.dim,
        "chunks": len(store.chunks),
        "recall": {k: recall[k] / total for k in KS},
        "mrr": sum(reciprocal_ranks) / total,
        "failures": failures,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure retrieval quality against labelled questions.",
    )
    parser.add_argument("backends", nargs="*", help="embedding backends to compare")
    parser.add_argument("--full", action="store_true", help="print wrong answers and analysis")
    args = parser.parse_args()

    specs = args.backends or DEFAULT_BACKENDS
    questions = load_questions()

    results = []
    for spec in specs:
        try:
            results.append(evaluate(spec, questions))
        except ImportError as exc:
            print(f"skipping {spec}: {exc}".split("\n")[0])
            print()

    if not results:
        return

    print("STEP 11 · Evaluate retrieval")
    print("-" * 74)
    print(f"{len(questions)} labelled questions across {len(DOCS)} documents")
    print()
    print(f"{'backend':<40} {'dim':>5} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
    print("-" * 74)
    for result in results:
        recall = result["recall"]
        print(f"{result['backend']:<40} {result['dim']:>5} "
              f"{recall[1]:>6.2f} {recall[3]:>6.2f} {recall[5]:>6.2f} "
              f"{result['mrr']:>6.2f}")
    print()

    if not args.full:
        best = max(results, key=lambda r: r["mrr"])
        print(f"Best backend here: {best['backend']}")
        print("Answer: if retrieval fails, RAG cannot answer correctly.")
        return

    # The comparison that matters: did anything beat the null model?
    floor = next((r for r in results if r["backend"].startswith("random")), None)
    if floor:
        print(f"Null model (random) recall@3 = {floor['recall'][3]:.2f} — that is")
        print("what zero knowledge of your corpus scores. Anything close to it")
        print("is not doing retrieval, it is doing decoration.")
        print()

    # Failures are where the learning is. Show them.
    best = max(results, key=lambda r: r["mrr"])
    print("=" * 74)
    print(f"WHERE {best['backend']} GOT IT WRONG "
          f"({len(best['failures'])} of {best['total']} not ranked first)")
    print("=" * 74)
    if not best["failures"]:
        print("  Nothing. On a 15-question set that means your set is too easy —")
        print("  add the questions your users actually ask badly.")
    for question, hits in best["failures"][:6]:
        top = hits[0] if hits else None
        print(f"  Q  {question['question']}")
        print(f"     wanted: {question['source']} · {question['section']}")
        if top:
            print(f"     got   : {top.chunk.source} · {top.chunk.section} "
                  f"({top.score:.3f})")
        print()

    print("Read those. Each one is a decision: is the chunking wrong, is the")
    print("question phrased in words the document never uses, or does the model")
    print("simply not understand the language? Three different fixes.")
    print()
    print("To measure a real model, download it first, then:")
    print("  python3 setup_check.py --download paraphrase-multilingual-MiniLM-L12-v2")
    print("  python3 11_evaluate.py random hashing st:paraphrase-multilingual-MiniLM-L12-v2")
    print()
    print("I am not going to tell you which embedding model is best for")
    print("Kinyarwanda, because I have not measured it and neither has anyone")
    print("who is telling you. Run this. Then you will know, for your corpus.")


if __name__ == "__main__":
    main()
