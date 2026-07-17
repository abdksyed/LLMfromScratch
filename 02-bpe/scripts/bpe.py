from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer


VOCAB_SIZE = 10_000


def faithful_units(text: str) -> int:
    total = 0
    in_word = False
    for character in text:
        is_word = unicodedata.category(character)[0] in {"L", "M", "N"}
        if is_word:
            if not in_word:
                total += 1
            in_word = True
        else:
            in_word = False
            if not character.isspace():
                total += 1
    return total


def visible_text(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


class SubmissionTokenizer:
    """Small evaluator-facing wrapper around the standard tokenizer artifact."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: Path) -> "SubmissionTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
        )


def verify(base_dir: Path) -> dict:
    stats = json.loads((base_dir / "stats.json").read_text(encoding="utf-8"))
    tokenizer = SubmissionTokenizer.from_file(base_dir / "tokenizer.json")
    if tokenizer.vocab_size != stats["vocab_size"] or tokenizer.vocab_size != VOCAB_SIZE:
        raise AssertionError("vocabulary size is not exactly 10,000")

    for language in stats["languages"]:
        path = base_dir / language["file"]
        text = path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        if decoded != text:
            raise AssertionError(f"{language['code']} failed exact round-trip")
        if visible_text(decoded) != visible_text(text):
            raise AssertionError(f"{language['code']} failed visible-text gate")
        if faithful_units(text) != language["faithful_units"]:
            raise AssertionError(f"{language['code']} faithful-unit count changed")
        if len(token_ids) != language["scored_tokens"]:
            raise AssertionError(f"{language['code']} token count changed")

    samples = (
        "India's population is 1,428,627,663.",
        "  leading\tspaces\nnew line  ",
        "भारत తెలుగు ಭಾರತ 🙂",
        "𓀀",
        "",
    )
    for sample in samples:
        decoded = tokenizer.decode(tokenizer.encode(sample))
        if decoded != sample:
            raise AssertionError(f"sample failed exact round-trip: {sample!r}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify the submitted tokenizer.")
    parser.add_argument("command", choices=("verify", "report"), nargs="?", default="report")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets",
    )
    args = parser.parse_args()
    stats = verify(args.base_dir) if args.command == "verify" else json.loads(
        (args.base_dir / "stats.json").read_text(encoding="utf-8")
    )
    print(f"vocab={stats['vocab_size']} score={stats['score_display']}")
    for row in sorted(stats["languages"], key=lambda item: item["scored_ratio"]):
        print(
            f"{row['code']}: {row['scored_tokens']}/{row['faithful_units']} "
            f"= {row['scored_ratio']:.9f}"
        )


if __name__ == "__main__":
    main()
