#!/usr/bin/env python3
"""
Split raw UTF-8 story text into one sentence per line for sandbox.py / load_corpus.

Typical workflow:
  python sandbox.py --print-vocab > vocab.txt
  python scripts/text_to_corpus_lines.py story_raw.txt -o data/my_stories.txt --allowlist vocab.txt

Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def load_allowlist(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                words.add(w)
    return words


def split_sentences(text: str) -> list[str]:
    """Rough sentence segmentation (no NLTK)."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def filter_sentence(
    sentence: str,
    allow: set[str] | None,
    min_w: int,
    max_w: int,
) -> str | None:
    words_raw = sentence.lower().split()
    if allow is not None:
        words = [w for w in words_raw if w in allow]
    else:
        words = words_raw
    n = len(words)
    if n < min_w or n > max_w:
        return None
    return " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert raw text to one sentence per line (for woke-baby-llm corpus files)."
    )
    ap.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="UTF-8 text file (omit to read stdin)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path (default: stdout)",
    )
    ap.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Drop sentences with fewer than this many words after filtering (default: 4).",
    )
    ap.add_argument(
        "--max-words",
        type=int,
        default=24,
        help="Drop sentences with more than this many words (default: 24).",
    )
    ap.add_argument(
        "--allowlist",
        type=Path,
        help="One word per line; only those words are kept. Build with: python sandbox.py --print-vocab",
    )
    args = ap.parse_args()
    if args.min_words < 1:
        raise SystemExit("--min-words must be >= 1")
    if args.max_words < args.min_words:
        raise SystemExit("--max-words must be >= --min-words")

    allow = load_allowlist(args.allowlist) if args.allowlist else None
    if args.input is not None:
        text = args.input.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    lines_out: list[str] = []
    for sent in split_sentences(text):
        line = filter_sentence(sent, allow, args.min_words, args.max_words)
        if line is not None:
            lines_out.append(line)

    body = "\n".join(lines_out)
    if lines_out:
        body += "\n"

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()
