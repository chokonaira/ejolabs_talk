"""
Run one lesson demo without remembering the file names.

Examples:

    python3 run_lesson.py list
    python3 run_lesson.py 1
    python3 run_lesson.py rag
    python3 run_lesson.py quick

Use `quick` for the shorter presenter flow:
without RAG -> with context -> RAG -> embeddings -> JSON -> streaming.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Lesson:
    key: str
    title: str
    command: tuple[str, ...]
    point: str


LESSONS = [
    Lesson(
        key="1",
        title="Ask without RAG",
        command=("01_ask_without_rag.py",),
        point="The model sounds confident but invents a phone number.",
    ),
    Lesson(
        key="2",
        title="Add document context",
        command=("02_add_context.py",),
        point="The same question becomes correct when the document is placed in the prompt.",
    ),
    Lesson(
        key="3",
        title="Structured output",
        command=("03_structured_output.py",),
        point="JSON fields are easier for code to validate and render than a paragraph.",
    ),
    Lesson(
        key="4",
        title="Streaming",
        command=("04_streaming.py",),
        point="Streaming does not finish sooner; it shows progress sooner.",
    ),
    Lesson(
        key="5",
        title="Chunking",
        command=("05_chunking.py",),
        point="Bad chunk boundaries can hide the answer from retrieval.",
    ),
    Lesson(
        key="6",
        title="Embeddings",
        command=("06_embeddings.py",),
        point="Embeddings turn text into vectors so nearby meanings can be compared.",
    ),
    Lesson(
        key="7",
        title="Retrieve",
        command=("07_retrieve.py",),
        point="Retrieval searches your own documents before the LLM writes anything.",
    ),
    Lesson(
        key="8",
        title="RAG",
        command=("08_rag.py",),
        point="RAG finds relevant chunks, puts them in the prompt, then asks the model.",
    ),
    Lesson(
        key="9",
        title="Evaluate retrieval",
        command=("11_evaluate.py",),
        point="Measure whether the right chunk appears in the top results.",
    ),
]

ALIASES = {
    "without-rag": "1",
    "context": "2",
    "json": "3",
    "structured": "3",
    "stream": "4",
    "streaming": "4",
    "chunk": "5",
    "chunking": "5",
    "embed": "6",
    "embedding": "6",
    "embeddings": "6",
    "retrieve": "7",
    "retrieval": "7",
    "rag": "8",
    "eval": "9",
    "evaluate": "9",
}

QUICK_FLOW = ("1", "2", "8", "6", "3", "4")
FULL_FLOW = tuple(lesson.key for lesson in LESSONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Session 5 lesson demos by number or name.",
    )
    parser.add_argument(
        "lesson",
        nargs="?",
        default="list",
        help="lesson number/name, list, quick, or all",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="do not wait between scripts when running quick or all",
    )
    args = parser.parse_args()

    target = args.lesson.lower()

    if target in {"list", "ls", "menu"}:
        print_menu()
        return

    if target == "quick":
        run_many(QUICK_FLOW, pause=not args.no_pause)
        return

    if target == "all":
        run_many(FULL_FLOW, pause=not args.no_pause)
        return

    run_one(resolve_key(target))


def print_menu() -> None:
    print()
    print("Session 5 lesson demos")
    print("-" * 72)
    for lesson in LESSONS:
        command = format_command(lesson.command)
        print(f"{lesson.key}. {lesson.title}")
        print(f"   Run:   {command}")
        print(f"   Point: {lesson.point}")
        print()
    print("Short presenter flow:")
    print("   python3 run_lesson.py quick")
    print()
    print("Everything:")
    print("   python3 run_lesson.py all")


def run_many(keys: tuple[str, ...], *, pause: bool) -> None:
    for index, key in enumerate(keys, start=1):
        lesson = find_lesson(key)
        print()
        print(f"{index}/{len(keys)} · {lesson.title}")
        print("-" * 72)
        run_one(key, show_title=False)

        if pause and index < len(keys):
            input("\nPress Enter for the next demo...")


def run_one(key: str, *, show_title: bool = True) -> None:
    lesson = find_lesson(key)
    if show_title:
        print(f"Lesson {lesson.key}: {lesson.title}")
        print()
    sys.stdout.flush()

    completed = subprocess.run(
        (sys.executable, *lesson.command),
        cwd=ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def resolve_key(value: str) -> str:
    if value in ALIASES:
        return ALIASES[value]
    return value


def find_lesson(key: str) -> Lesson:
    for lesson in LESSONS:
        if lesson.key == key:
            return lesson

    print(f"Unknown lesson: {key}")
    print()
    print_menu()
    raise SystemExit(2)


def format_command(command: tuple[str, ...]) -> str:
    return "python3 " + " ".join(command)


if __name__ == "__main__":
    main()
