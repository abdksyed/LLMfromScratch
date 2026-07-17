from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

from build_faithful_corpus import faithful_units


VOCAB_SIZE = 10_000
BYTE_FALLBACK_SIZE = 256
TRAINED_VOCAB_SIZE = VOCAB_SIZE - BYTE_FALLBACK_SIZE
LANGUAGES = ("en", "hi", "te", "kn")
LANGUAGE_SCRIPTS = {
    "en": "Latin",
    "hi": "Devanagari",
    "te": "Telugu",
    "kn": "Kannada",
}
TRAINING_WEIGHTS = {"en": 15, "hi": 19, "te": 32, "kn": 67}
ROUNDTRIP_SAMPLES = (
    "India's population is 1,428,627,663.",
    "  leading\tspaces\nnew line  ",
    "भारत తెలుగు ಭಾರತ 🙂",
    "𓀀",
    "",
)


def corpus_paths(base_dir: Path) -> dict[str, Path]:
    return {
        code: base_dir / f"wikipedia-{code}.txt"
        for code in LANGUAGES
    }


def parse_merges(raw_merges: list[str | list[str]]) -> list[tuple[str, str]]:
    return [
        tuple(merge.split(" ", 1)) if isinstance(merge, str) else tuple(merge)
        for merge in raw_merges
    ]


def make_tokenizer(vocab: dict[str, int], merges: list[tuple[str, str]]) -> Tokenizer:
    tokenizer = Tokenizer(
        BPE(
            vocab=vocab,
            merges=merges,
            unk_token="[UNK]",
            byte_fallback=True,
        )
    )
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tokenizer.decoder = DecoderSequence(
        [
            ByteFallback(),
            MetaspaceDecoder(replacement="▁", prepend_scheme="never"),
        ]
    )
    return tokenizer


def train_tokenizer(base_dir: Path) -> Tokenizer:
    paths = corpus_paths(base_dir)
    training_files = [
        str(paths[code])
        for code in LANGUAGES
        for _ in range(TRAINING_WEIGHTS[code])
    ]

    base = Tokenizer(BPE(unk_token="[UNK]"))
    base.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    base.train(
        training_files,
        BpeTrainer(
            vocab_size=TRAINED_VOCAB_SIZE,
            min_frequency=1,
            special_tokens=["[UNK]"],
        ),
    )

    model = json.loads(base.to_str())["model"]
    vocab = model["vocab"]
    if len(vocab) != TRAINED_VOCAB_SIZE:
        raise RuntimeError(
            f"trainer produced {len(vocab)} entries, expected {TRAINED_VOCAB_SIZE}"
        )

    for value in range(BYTE_FALLBACK_SIZE):
        token = f"<0x{value:02X}>"
        if token in vocab:
            raise RuntimeError(f"trained vocabulary collides with {token}")
        vocab[token] = len(vocab)

    tokenizer = make_tokenizer(vocab, parse_merges(model["merges"]))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError("final tokenizer does not contain exactly 10,000 tokens")
    return tokenizer


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_text(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def evaluate(
    tokenizer: Tokenizer,
    base_dir: Path,
    source_data: dict[str, Any],
) -> dict[str, Any]:
    sources = {page["code"]: page for page in source_data["pages"]}
    rows: list[dict[str, Any]] = []

    for code, path in corpus_paths(base_dir).items():
        source = sources[code]
        text = path.read_text(encoding="utf-8")
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        if decoded != text:
            raise AssertionError(f"{code} corpus failed exact round-trip")

        units = faithful_units(text)
        rows.append(
            {
                **source,
                "script": LANGUAGE_SCRIPTS[code],
                "sha256": file_sha256(path),
                "bytes": len(text.encode("utf-8")),
                "characters": len(text),
                "words": len(text.split()),
                "faithful_units": units,
                "scored_tokens": len(encoding.ids),
                "scored_ratio": len(encoding.ids) / units,
                "unknown_tokens_emitted": encoding.ids.count(
                    tokenizer.token_to_id("[UNK]")
                ),
                "exact_roundtrip": True,
            }
        )

    ratios = [row["scored_ratio"] for row in rows]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread
    hindi_ratio = next(row["scored_ratio"] for row in rows if row["code"] == "hi")
    hindi_penalty = math.exp(max(0.0, hindi_ratio / 1.2 - 1.0))

    model = json.loads(tokenizer.to_str())["model"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": "wiki_faithful_markdown",
        "tokenizer_format": "Hugging Face Tokenizers BPE",
        "vocab_size": tokenizer.get_vocab_size(),
        "vocab_composition": {
            "trained_entries": TRAINED_VOCAB_SIZE,
            "byte_fallback_entries": BYTE_FALLBACK_SIZE,
            "merges": len(model["merges"]),
        },
        "training_weights": TRAINING_WEIGHTS,
        "fourth_language": "Kannada",
        "unit_policy": source_data["unit_policy"],
        "ratio_limit": 1.2,
        "score_formula": "1000 / (max(X) - min(X))",
        "spread": spread,
        "score": score,
        "score_display": f"{score:.2f}",
        "hindi_penalty_factor": hindi_penalty,
        "hindi_adjusted_score": score / hindi_penalty,
        "lossless": True,
        "exact_roundtrip": True,
        "unknown_tokens_emitted": sum(
            row["unknown_tokens_emitted"] for row in rows
        ),
        "languages": rows,
    }


def verify_roundtrips(tokenizer: Tokenizer) -> None:
    for sample in ROUNDTRIP_SAMPLES:
        ids = tokenizer.encode(sample).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != sample:
            raise AssertionError(
                f"round-trip failed: {sample!r} decoded as {decoded!r}"
            )
        if visible_text(decoded) != visible_text(sample):
            raise AssertionError(f"visible-text gate failed for {sample!r}")


def build(base_dir: Path) -> dict[str, Any]:
    source_path = base_dir / "wikipedia-sources.json"
    source_data = json.loads(source_path.read_text(encoding="utf-8"))
    tokenizer = train_tokenizer(base_dir)
    verify_roundtrips(tokenizer)
    stats = evaluate(tokenizer, base_dir, source_data)
    tokenizer.save(str(base_dir / "tokenizer.json"), pretty=True)
    (base_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def verify(base_dir: Path) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(base_dir / "tokenizer.json"))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise AssertionError("tokenizer does not contain exactly 10,000 tokens")
    verify_roundtrips(tokenizer)

    source_data = json.loads(
        (base_dir / "wikipedia-sources.json").read_text(encoding="utf-8")
    )
    actual = evaluate(tokenizer, base_dir, source_data)
    expected = json.loads((base_dir / "stats.json").read_text(encoding="utf-8"))
    for field in (
        "vocab_size",
        "spread",
        "score",
        "unknown_tokens_emitted",
        "exact_roundtrip",
    ):
        if actual[field] != expected[field]:
            raise AssertionError(f"saved {field} does not reproduce")

    expected_rows = {row["code"]: row for row in expected["languages"]}
    for row in actual["languages"]:
        saved = expected_rows[row["code"]]
        for field in ("sha256", "faithful_units", "scored_tokens", "scored_ratio"):
            if row[field] != saved[field]:
                raise AssertionError(f"{row['code']} {field} does not reproduce")
        if row["scored_ratio"] > expected["ratio_limit"]:
            raise AssertionError(f"{row['code']} exceeds the 1.2 ratio limit")
    return expected


def print_report(stats: dict[str, Any]) -> None:
    print(f"vocab={stats['vocab_size']} score={stats['score_display']}")
    print(f"spread={stats['spread']:.12f} exact_roundtrip={stats['exact_roundtrip']}")
    for row in sorted(stats["languages"], key=lambda item: item["scored_ratio"]):
        print(
            f"{row['code']}: {row['scored_tokens']}/{row['faithful_units']} "
            f"= {row['scored_ratio']:.9f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the 10k tokenizer.")
    parser.add_argument("command", choices=("build", "verify", "report"), nargs="?", default="report")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets",
    )
    args = parser.parse_args()

    if args.command == "build":
        stats = build(args.base_dir)
    elif args.command == "verify":
        stats = verify(args.base_dir)
    else:
        stats = json.loads((args.base_dir / "stats.json").read_text(encoding="utf-8"))
    print_report(stats)


if __name__ == "__main__":
    main()
