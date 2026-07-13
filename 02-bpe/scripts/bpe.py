from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"\S+")


def split_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def read_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class SubmissionTokenizer:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.tokens = spec["tokens"]
        self.id_to_bytes = {
            token["id"]: bytes.fromhex(token["hex"]) for token in self.tokens
        }
        self.merges = {
            tuple(token["parents"]): token["id"]
            for token in self.tokens
            if token["kind"] == "merge"
        }
        self.lexemes: dict[str, int] = {}
        pieces: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)

        for token in self.tokens:
            if token["kind"] == "lexeme":
                self.lexemes[token["text"]] = token["id"]
            elif token["kind"] == "piece":
                pieces[token["source"]].append((token["piece_index"], token["id"]))

        self.pieces = {
            key: [token_id for _, token_id in sorted(parts)]
            for key, parts in pieces.items()
        }

    @classmethod
    def from_file(cls, path: Path) -> "SubmissionTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _merge(
        sequence: tuple[int, ...], pair: tuple[int, int], token_id: int
    ) -> tuple[int, ...]:
        output: list[int] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and sequence[index : index + 2] == pair:
                output.append(token_id)
                index += 2
            else:
                output.append(sequence[index])
                index += 1
        return tuple(output)

    def encode_bpe_word(self, word: str) -> list[int]:
        sequence = tuple(word.encode("utf-8"))
        while len(sequence) > 1:
            ranked = [
                (self.merges[pair], pair)
                for pair in zip(sequence, sequence[1:])
                if pair in self.merges
            ]
            if not ranked:
                break
            token_id, pair = min(ranked)
            sequence = self._merge(sequence, pair, token_id)
        return list(sequence)

    def encode_word(self, word: str) -> list[int]:
        if word in self.lexemes:
            return [self.lexemes[word]]
        if word in self.pieces:
            return self.pieces[word]
        return self.encode_bpe_word(word)

    def encode(self, text: str) -> list[int]:
        return [
            token_id
            for word in split_words(text)
            for token_id in self.encode_word(word)
        ]

    def decode_word(self, token_ids: list[int]) -> str:
        return b"".join(self.id_to_bytes[token_id] for token_id in token_ids).decode(
            "utf-8"
        )


def verify(base_dir: Path) -> dict[str, Any]:
    stats = json.loads((base_dir / "stats.json").read_text(encoding="utf-8"))
    tokenizer = SubmissionTokenizer.from_file(base_dir / "tokenizer.json")
    assert len(tokenizer.tokens) == stats["vocab_size"] == 10_000
    assert [token["id"] for token in tokenizer.tokens] == list(range(10_000))

    for language in stats["languages"]:
        path = base_dir / language["file"]
        text = read_corpus(path)
        words = split_words(text)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == language["sha256"]
        assert len(words) == language["words"]
        assert len(tokenizer.encode(text)) == language["scored_tokens"]
        for word in set(words):
            token_ids = tokenizer.encode_word(word)
            assert tokenizer.decode_word(token_ids) == word
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify the submitted tokenizer.")
    parser.add_argument("command", choices=("verify", "report"), nargs="?", default="report")
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).resolve().parent.parent / "assets"
    )
    args = parser.parse_args()
    stats = verify(args.base_dir)
    print(f"vocab={stats['vocab_size']} score={stats['score_display']}")
    for row in sorted(stats["languages"], key=lambda item: item["scored_ratio"]):
        print(
            f"{row['code']}: {row['scored_tokens']}/{row['words']} "
            f"= {row['scored_ratio']:.9f}"
        )


if __name__ == "__main__":
    main()
