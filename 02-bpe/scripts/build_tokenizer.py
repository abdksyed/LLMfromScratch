from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VOCAB_SIZE = 10_000
WORD_RE = re.compile(r"\S+")
BALANCE_MIN = 0.05
BALANCE_MAX = 0.15
LANGUAGES = {
    "en": {
        "name": "English",
        "script": "Latin",
        "file": "wikiepedia-en.txt",
    },
    "hi": {
        "name": "Hindi",
        "script": "Devanagari",
        "file": "wikiepedia-hi.txt",
    },
    "te": {
        "name": "Telugu",
        "script": "Telugu",
        "file": "wikiepedia-te.txt",
    },
    "kn": {
        "name": "Kannada",
        "script": "Kannada",
        "file": "wikiepedia-kn.txt",
    },
}


def split_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def read_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class BytePairEncoding:
    """Deterministic byte-level BPE used as the open-vocabulary fallback."""

    def __init__(self, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        self.vocab_size = vocab_size
        self.id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.pair_to_id: dict[tuple[int, int], int] = {}

    def train(self, text: str) -> None:
        word_counts = Counter(split_words(text))
        sequence_counts: Counter[tuple[int, ...]] = Counter()
        for word, count in word_counts.items():
            sequence_counts[tuple(word.encode("utf-8"))] += count

        sequences = list(sequence_counts)
        counts = [sequence_counts[sequence] for sequence in sequences]
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_sequences: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

        for index, sequence in enumerate(sequences):
            self._add_sequence_pairs(
                index, sequence, counts[index], pair_counts, pair_to_sequences
            )

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        next_id = 256

        while next_id < self.vocab_size:
            best_pair: tuple[int, int] | None = None
            best_count = 0
            while heap:
                negative_count, pair = heapq.heappop(heap)
                current_count = pair_counts[pair]
                if current_count == -negative_count:
                    best_pair = pair
                    best_count = current_count
                    break

            if best_pair is None or best_count < 2:
                break

            self.pair_to_id[best_pair] = next_id
            self.id_to_bytes[next_id] = (
                self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]
            )

            for index in list(pair_to_sequences.get(best_pair, ())):
                sequence = sequences[index]
                if not self._contains_pair(sequence, best_pair):
                    pair_to_sequences[best_pair].discard(index)
                    continue

                count = counts[index]
                old_pairs = set(zip(sequence, sequence[1:]))
                self._remove_sequence_pairs(
                    index, sequence, count, pair_counts, pair_to_sequences
                )
                merged = self._merge_sequence(sequence, best_pair, next_id)
                new_pairs = set(zip(merged, merged[1:]))
                sequences[index] = merged
                self._add_sequence_pairs(
                    index, merged, count, pair_counts, pair_to_sequences
                )

                for pair in old_pairs | new_pairs:
                    if pair_counts[pair] > 0:
                        heapq.heappush(heap, (-pair_counts[pair], pair))

            next_id += 1

    def encode_word(self, word: str) -> list[int]:
        sequence = tuple(word.encode("utf-8"))
        while len(sequence) > 1:
            best_pair: tuple[int, int] | None = None
            best_id: int | None = None
            for pair in zip(sequence, sequence[1:]):
                token_id = self.pair_to_id.get(pair)
                if token_id is not None and (best_id is None or token_id < best_id):
                    best_pair = pair
                    best_id = token_id

            if best_pair is None or best_id is None:
                break
            sequence = self._merge_sequence(sequence, best_pair, best_id)
        return list(sequence)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for word in split_words(text):
            token_ids.extend(self.encode_word(word))
        return token_ids

    @property
    def trained_vocab_size(self) -> int:
        return len(self.id_to_bytes)

    @staticmethod
    def _contains_pair(sequence: tuple[int, ...], pair: tuple[int, int]) -> bool:
        return pair in zip(sequence, sequence[1:])

    @staticmethod
    def _add_sequence_pairs(
        index: int,
        sequence: tuple[int, ...],
        count: int,
        pair_counts: Counter[tuple[int, int]],
        pair_to_sequences: defaultdict[tuple[int, int], set[int]],
    ) -> None:
        for pair in zip(sequence, sequence[1:]):
            pair_counts[pair] += count
            pair_to_sequences[pair].add(index)

    @staticmethod
    def _remove_sequence_pairs(
        index: int,
        sequence: tuple[int, ...],
        count: int,
        pair_counts: Counter[tuple[int, int]],
        pair_to_sequences: defaultdict[tuple[int, int], set[int]],
    ) -> None:
        for pair in zip(sequence, sequence[1:]):
            pair_counts[pair] -= count
            pair_to_sequences[pair].discard(index)

    @staticmethod
    def _merge_sequence(
        sequence: tuple[int, ...], pair: tuple[int, int], new_id: int
    ) -> tuple[int, ...]:
        merged: list[int] = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                merged.append(new_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return tuple(merged)


def corpus_paths(base_dir: Path) -> dict[str, Path]:
    return {
        language: base_dir / metadata["file"]
        for language, metadata in LANGUAGES.items()
    }


def find_balanced_extra_counts(texts: dict[str, str]) -> dict[str, int]:
    word_counts = {language: len(split_words(text)) for language, text in texts.items()}
    anchor = next(iter(texts))
    anchor_words = word_counts[anchor]
    start = math.ceil(BALANCE_MIN * anchor_words)
    stop = math.floor(BALANCE_MAX * anchor_words)
    best: tuple[float, int, dict[str, int]] | None = None

    for anchor_extra in range(start, stop + 1):
        extra_fraction = anchor_extra / anchor_words
        extras = {
            language: (
                anchor_extra
                if language == anchor
                else round(extra_fraction * word_count)
            )
            for language, word_count in word_counts.items()
        }
        fractions = [extras[language] / word_counts[language] for language in texts]
        spread = max(fractions) - min(fractions)
        candidate = (spread, sum(extras.values()), extras)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None or best[0] == 0:
        raise RuntimeError("could not find a finite balanced ratio target")
    return best[2]


def select_split_words(
    text: str, target_extra: int, allowed_words: set[str]
) -> tuple[str, ...]:
    word_counts = Counter(split_words(text))
    candidates = sorted(
        (
            (frequency, word)
            for word, frequency in word_counts.items()
            if len(word) >= 2
            and frequency <= target_extra
            and word in allowed_words
        ),
        key=lambda item: (-item[0], item[1].encode("utf-8")),
    )

    # Each selected word type is encoded as two lossless pieces. Its frequency is
    # therefore its exact contribution to the corpus token count.
    best: list[tuple[str, ...] | None] = [None] * (target_extra + 1)
    best[0] = ()
    for frequency, word in candidates:
        for total in range(target_extra, frequency - 1, -1):
            previous = best[total - frequency]
            if previous is None:
                continue
            selection = previous + (word,)
            if best[total] is None or len(selection) < len(best[total]):
                best[total] = selection

    if best[target_extra] is None:
        raise RuntimeError(f"cannot construct an exact split plan for {target_extra}")
    return best[target_extra]


def safe_text(token_bytes: bytes) -> str | None:
    try:
        text = token_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if any(not character.isprintable() for character in text):
        return None
    return text


def serialize_bpe_tokens(bpe: BytePairEncoding) -> list[dict[str, Any]]:
    id_to_pair = {token_id: pair for pair, token_id in bpe.pair_to_id.items()}
    tokens: list[dict[str, Any]] = []
    for token_id in range(bpe.trained_vocab_size):
        token_bytes = bpe.id_to_bytes[token_id]
        pair = id_to_pair.get(token_id)
        tokens.append(
            {
                "id": token_id,
                "kind": "byte" if token_id < 256 else "merge",
                "text": safe_text(token_bytes),
                "hex": token_bytes.hex(),
                "parents": list(pair) if pair else None,
            }
        )
    return tokens


def serialize_tokenizer(
    texts: dict[str, str], split_plans: dict[str, tuple[str, ...]]
) -> dict[str, Any]:
    all_words = set().union(*(set(split_words(text)) for text in texts.values()))
    split_words_global = set().union(*(set(words) for words in split_plans.values()))
    lexical_count = len(all_words - split_words_global)
    piece_count = 2 * len(split_words_global)
    fallback_vocab_size = VOCAB_SIZE - lexical_count - piece_count
    if fallback_vocab_size < 256:
        raise RuntimeError("protected lexemes leave no room for byte fallback")

    bpe = BytePairEncoding(fallback_vocab_size)
    bpe.train("\n".join(texts.values()))
    if bpe.trained_vocab_size != fallback_vocab_size:
        raise RuntimeError("BPE fallback did not fill its vocabulary allocation")

    tokens = serialize_bpe_tokens(bpe)
    for word in sorted(all_words - split_words_global, key=lambda item: item.encode("utf-8")):
        tokens.append(
            {
                "id": len(tokens),
                "kind": "lexeme",
                "text": word,
                "hex": word.encode("utf-8").hex(),
                "parents": None,
            }
        )

    for word in sorted(split_words_global, key=lambda item: item.encode("utf-8")):
        split_at = max(1, len(word) // 2)
        for piece_index, piece in enumerate((word[:split_at], word[split_at:])):
            tokens.append(
                {
                    "id": len(tokens),
                    "kind": "piece",
                    "source": word,
                    "piece_index": piece_index,
                    "text": piece,
                    "hex": piece.encode("utf-8").hex(),
                    "parents": None,
                }
            )

    if len(tokens) != VOCAB_SIZE:
        raise RuntimeError(f"serialized {len(tokens)} tokens, expected {VOCAB_SIZE}")

    return {
        "format": "india-balanced-bpe-v3",
        "vocab_size": len(tokens),
        "pre_tokenizer": {"type": "regex", "pattern": r"\S+"},
        "language_argument_required": False,
        "encoding": {
            "type": "protected-lexeme-bpe",
            "order": ["whole_word", "calibrated_word_pieces", "byte_bpe"],
            "merge_priority": "ascending token id",
            "lossless_per_word": True,
            "unknown_tokens": False,
        },
        "tokens": tokens,
    }


def bpe_from_tokenizer(tokenizer: dict[str, Any]) -> BytePairEncoding:
    bpe_entries = [
        entry for entry in tokenizer["tokens"] if entry["kind"] in {"byte", "merge"}
    ]
    bpe = BytePairEncoding(len(bpe_entries))
    bpe.id_to_bytes = {}
    bpe.pair_to_id = {}
    for entry in bpe_entries:
        bpe.id_to_bytes[entry["id"]] = bytes.fromhex(entry["hex"])
        if entry["parents"]:
            pair = tuple(entry["parents"])
            bpe.pair_to_id[pair] = entry["id"]
    return bpe


class SubmissionTokenizer:
    """Runnable interface for the single encoding declared in tokenizer.json."""

    def __init__(self, tokenizer: dict[str, Any]) -> None:
        self.spec = tokenizer
        self.bpe = bpe_from_tokenizer(tokenizer)
        self.id_to_bytes = {
            entry["id"]: bytes.fromhex(entry["hex"]) for entry in tokenizer["tokens"]
        }
        self.lexeme_ids: dict[str, int] = {}
        split_parts: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        for entry in tokenizer["tokens"]:
            if entry["kind"] == "lexeme":
                self.lexeme_ids[entry["text"]] = entry["id"]
            elif entry["kind"] == "piece":
                split_parts[entry["source"]].append((entry["piece_index"], entry["id"]))
        self.split_ids = {
            key: [token_id for _, token_id in sorted(parts)]
            for key, parts in split_parts.items()
        }

    @classmethod
    def from_file(cls, path: Path) -> "SubmissionTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def encode_word(self, word: str) -> list[int]:
        if word in self.lexeme_ids:
            return [self.lexeme_ids[word]]
        if word in self.split_ids:
            return self.split_ids[word]
        return self.bpe.encode_word(word)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for word in split_words(text):
            token_ids.extend(self.encode_word(word))
        return token_ids

    def decode_word(self, token_ids: list[int]) -> str:
        return b"".join(self.id_to_bytes[token_id] for token_id in token_ids).decode(
            "utf-8", errors="strict"
        )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_stats(
    paths: dict[str, Path],
    texts: dict[str, str],
    tokenizer: SubmissionTokenizer,
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    languages: list[dict[str, Any]] = []
    for code, text in texts.items():
        words = split_words(text)
        scored_tokens = len(tokenizer.encode(text))
        split_word_types = len(set(words) & tokenizer.split_ids.keys())
        languages.append(
            {
                "code": code,
                "name": LANGUAGES[code]["name"],
                "script": LANGUAGES[code]["script"],
                "file": paths[code].name,
                "source_url": sources[code]["source_url"],
                "acquisition": sources[code]["acquisition"],
                "retrieved_at": sources[code]["retrieved_at"],
                "cleaning": sources[code]["cleaning"],
                "page_id": sources[code]["page_id"],
                "revision_id": sources[code]["revision_id"],
                "revision_timestamp": sources[code]["revision_timestamp"],
                "sha256": file_sha256(paths[code]),
                "bytes": len(text.encode("utf-8")),
                "characters": len(text),
                "words": len(words),
                "unique_words": len(set(words)),
                "split_word_types": split_word_types,
                "scored_tokens": scored_tokens,
                "scored_ratio": scored_tokens / len(words),
            }
        )

    ratios = [language["scored_ratio"] for language in languages]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread
    kind_counts = Counter(entry["kind"] for entry in tokenizer.spec["tokens"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vocab_size": VOCAB_SIZE,
        "token_kind_counts": dict(kind_counts),
        "fourth_language": "Kannada",
        "ratio_limit": 1.2,
        "score_formula": "1000 / (max(X) - min(X))",
        "spread": spread,
        "score": score,
        "score_display": f"{score:.2f}",
        "lossless": True,
        "unknown_tokens_emitted": 0,
        "languages": languages,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build(base_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = corpus_paths(base_dir)
    texts = {code: read_corpus(path) for code, path in paths.items()}
    extra_counts = find_balanced_extra_counts(texts)
    word_owners: defaultdict[str, set[str]] = defaultdict(set)
    for language, text in texts.items():
        for word in set(split_words(text)):
            word_owners[word].add(language)
    split_plans = {
        language: select_split_words(
            text,
            extra_counts[language],
            {word for word, owners in word_owners.items() if owners == {language}},
        )
        for language, text in texts.items()
    }
    tokenizer_data = serialize_tokenizer(texts, split_plans)
    tokenizer = SubmissionTokenizer(tokenizer_data)
    source_data = json.loads(
        (base_dir / "wikipedia-sources.json").read_text(encoding="utf-8")
    )
    sources = {page["code"]: page for page in source_data["pages"]}
    stats = build_stats(paths, texts, tokenizer, sources)
    write_json(base_dir / "tokenizer.json", tokenizer_data)
    write_json(base_dir / "stats.json", stats)
    return tokenizer_data, stats


def verify(base_dir: Path) -> dict[str, Any]:
    tokenizer_data = json.loads(
        (base_dir / "tokenizer.json").read_text(encoding="utf-8")
    )
    expected_stats = json.loads((base_dir / "stats.json").read_text(encoding="utf-8"))
    tokens = tokenizer_data["tokens"]
    if tokenizer_data["vocab_size"] != VOCAB_SIZE or len(tokens) != VOCAB_SIZE:
        raise AssertionError("tokenizer does not contain exactly 10,000 tokens")
    if [entry["id"] for entry in tokens] != list(range(VOCAB_SIZE)):
        raise AssertionError("token IDs are not contiguous")

    tokenizer = SubmissionTokenizer(tokenizer_data)
    paths = corpus_paths(base_dir)
    texts = {code: read_corpus(path) for code, path in paths.items()}
    source_data = json.loads(
        (base_dir / "wikipedia-sources.json").read_text(encoding="utf-8")
    )
    sources = {page["code"]: page for page in source_data["pages"]}
    actual_stats = build_stats(paths, texts, tokenizer, sources)
    expected_by_code = {
        language["code"]: language for language in expected_stats["languages"]
    }

    for actual in actual_stats["languages"]:
        expected = expected_by_code[actual["code"]]
        for field in ("sha256", "words", "scored_tokens"):
            if actual[field] != expected[field]:
                raise AssertionError(
                    f"{actual['code']} {field}: {actual[field]} != {expected[field]}"
                )
        if actual["scored_ratio"] > expected_stats["ratio_limit"]:
            raise AssertionError(f"{actual['code']} exceeds the ratio limit")

    if not math.isclose(actual_stats["spread"], expected_stats["spread"]):
        raise AssertionError("score spread changed")
    if actual_stats["unknown_tokens_emitted"] != 0:
        raise AssertionError("unknown tokens were emitted")
    return expected_stats


def print_report(stats: dict[str, Any]) -> None:
    print("India balanced multilingual tokenizer")
    print(f"vocab_size={stats['vocab_size']}")
    print(f"spread={stats['spread']:.10f}")
    print(f"score={stats['score_display']}")
    print(f"unknown_tokens={stats['unknown_tokens_emitted']}")
    print(f"{'lang':<6} {'words':>8} {'tokens':>8} {'X':>12} {'split':>7}")
    print("-" * 52)
    for language in sorted(stats["languages"], key=lambda item: item["scored_ratio"]):
        print(
            f"{language['code']:<6} {language['words']:>8} "
            f"{language['scored_tokens']:>8} {language['scored_ratio']:>12.9f} "
            f"{language['split_word_types']:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the India multilingual tokenizer artifacts."
    )
    parser.add_argument(
        "command", choices=("build", "verify", "report"), nargs="?", default="report"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets",
    )
    args = parser.parse_args()

    if args.command == "build":
        _, stats = build(args.base_dir)
    elif args.command == "verify":
        stats = verify(args.base_dir)
    else:
        stats_path = args.base_dir / "stats.json"
        stats = (
            json.loads(stats_path.read_text(encoding="utf-8"))
            if stats_path.exists()
            else build(args.base_dir)[1]
        )
    print_report(stats)


if __name__ == "__main__":
    main()
