"""
PRE-FLIGHT — run this at home, the night before, on a connection you trust.

    python3 setup_check.py
    python3 setup_check.py --download paraphrase-multilingual-MiniLM-L12-v2

It checks that the pipeline runs, tells you exactly what is missing, and never
requires anything to be missing. A pass with zero optional extras installed is
a pass: every numbered script works offline with no API key.

The one thing you should NOT leave until the morning of the session is the
embedding model download. MiniLM is about 120 MB and LaBSE about 1.8 GB.
Twenty-five people pulling that over one room's wifi does not work.
"""

import argparse
import sys


GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def ok(message: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {message}")


def warn(message: str) -> None:
    print(f"  {YELLOW}----{RESET}  {message}")


def fail(message: str) -> None:
    print(f"  {RED}FAIL{RESET}  {message}")


def check_python() -> bool:
    version = sys.version_info
    text = f"Python {version.major}.{version.minor}.{version.micro}"
    if version >= (3, 10):
        ok(text)
        return True
    fail(f"{text} — this code needs 3.10 or newer (it uses `str | None` syntax)")
    return False


def check_pipeline() -> bool:
    """The real test: chunk, embed, retrieve, and get the right answer back."""
    try:
        from ejo.embed import get_embedder
        from ejo.store import build_store
    except Exception as exc:
        fail(f"cannot import the ejo package: {exc}")
        print(f"        {DIM}are you running this from inside the code/ directory?{RESET}")
        return False

    try:
        embedder = get_embedder("hashing")
        store = build_store(
            ["docs/kanombe-clinic.md",
             "docs/kanombe-fees.md",
             "docs/kanombe-services.md"],
            embedder, chunk_size=500, overlap=50,
        )
    except FileNotFoundError as exc:
        fail(f"cannot read the documents: {exc}")
        print(f"        {DIM}run this from inside the code/ directory{RESET}")
        return False

    hits = store.search("What number do I call at the weekend?", k=1)
    if hits and "0788 123 456" in hits[0].chunk.text:
        ok(f"offline pipeline works — {len(store.chunks)} chunks indexed, "
           f"right chunk retrieved at {hits[0].score:.3f}")
        return True
    fail("pipeline ran but retrieved the wrong chunk — tell the instructor")
    return False


def check_optional() -> None:
    for module, note in [
        ("requests", "needed only to call the live EjoChat API"),
        ("sentence_transformers", "needed only for real embedding models"),
        ("chromadb", "needed only for step 9"),
        ("psycopg", "needed only for step 9"),
    ]:
        try:
            __import__(module)
            ok(f"{module} installed")
        except ImportError:
            warn(f"{module} not installed — {note}")


def check_api_key() -> None:
    import os

    key = os.environ.get("EJO_API_KEY", "")
    if not key:
        warn("EJO_API_KEY not set — every script will use the offline mock, "
             "which is fine")
        return
    if not key.startswith("kgpt_"):
        warn(f"EJO_API_KEY is set but does not look like a key "
             f"(expected it to start with 'kgpt_')")
        return
    ok(f"EJO_API_KEY is set ({key[:9]}…) — remember the free tier is "
       f"50 requests/day")


def download(model_name: str) -> None:
    print(f"Downloading {model_name}. This is the slow part — do it on wifi you trust.")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        fail("sentence-transformers is not installed")
        print("        pip install sentence-transformers")
        return

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    ok(f"{model_name} ready — {dim} dimensions, cached locally")
    print()
    print("Now measure it against the baseline rather than trusting it:")
    print(f"  python3 11_evaluate.py random hashing st:{model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", metavar="MODEL",
                        help="download a sentence-transformers model and cache it")
    args = parser.parse_args()

    if args.download:
        download(args.download)
        return

    print()
    print("REQUIRED — nothing below this line needs the internet")
    print("-" * 62)
    required = [check_python(), check_pipeline()]

    print()
    print("OPTIONAL — everything works without these")
    print("-" * 62)
    check_optional()
    check_api_key()

    print()
    if all(required):
        print(f"{GREEN}Ready.{RESET} You can run every numbered script right now:")
        print("  python3 01_ask_without_rag.py")
        print("  python3 08_rag.py")
        print("  python3 11_evaluate.py")
        print()
        print("Before the session, if you can, also run:")
        print("  python3 setup_check.py --download paraphrase-multilingual-MiniLM-L12-v2")
    else:
        print(f"{RED}Not ready.{RESET} Fix the FAIL lines above, then run this again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
