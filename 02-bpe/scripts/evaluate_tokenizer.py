#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import regex
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
TOKENIZER_PATH = ASSETS / "tokenizer.json"
LANGUAGES = ("en", "hi", "te", "kn")
FAITHFUL_UNIT_RE = regex.compile(
    r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]"
)


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def visible_text(text: str) -> str:
    return regex.sub(r"\s+", "", text)


def evaluate() -> dict:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    if tokenizer.get_vocab_size() != 10_000:
        raise AssertionError(
            f"expected 10,000 vocabulary entries, got {tokenizer.get_vocab_size()}"
        )

    rows = {}
    for code in LANGUAGES:
        text = (ASSETS / f"wikipedia-{code}.txt").read_text(encoding="utf-8")
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        if visible_text(decoded) != visible_text(text):
            raise AssertionError(f"{code} failed the visible non-whitespace gate")
        if decoded != text:
            raise AssertionError(f"{code} failed the stricter exact round-trip gate")

        units = faithful_units(text)
        ratio = len(encoding.ids) / units
        if ratio > 1.2:
            raise AssertionError(f"{code} ratio {ratio:.9f} exceeds 1.2")
        rows[code] = {
            "tokens": len(encoding.ids),
            "faithful_units": units,
            "ratio": ratio,
            "exact_roundtrip": True,
        }

    ratios = [row["ratio"] for row in rows.values()]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, rows["hi"]["ratio"] / 1.2 - 1.0))
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "rows": rows,
        "spread": spread,
        "score": score,
        "hindi_exp1_penalty_factor": hindi_penalty,
        "hindi_exp1_adjusted_score": score / hindi_penalty,
        "faithfulness_gate": "PASS",
    }


def main() -> int:
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
