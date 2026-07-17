from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpe
import build_tokenizer


BASE_DIR = Path(__file__).resolve().parent.parent / "assets"


class TokenizerArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = json.loads(
            (BASE_DIR / "tokenizer.json").read_text(encoding="utf-8")
        )
        cls.stats = json.loads((BASE_DIR / "stats.json").read_text(encoding="utf-8"))
        cls.submission = bpe.SubmissionTokenizer.from_file(BASE_DIR / "tokenizer.json")

    def test_standard_vocabulary_has_exactly_ten_thousand_ids(self) -> None:
        vocab = self.artifact["model"]["vocab"]
        self.assertEqual(self.artifact["model"]["type"], "BPE")
        self.assertEqual(len(vocab), 10_000)
        self.assertEqual(set(vocab.values()), set(range(10_000)))
        self.assertEqual(self.submission.vocab_size, 10_000)
        self.assertIsNone(self.artifact["normalizer"])
        self.assertEqual(self.artifact["pre_tokenizer"]["type"], "Metaspace")
        self.assertTrue(self.artifact["model"]["byte_fallback"])
        for value in range(256):
            self.assertIn(f"<0x{value:02X}>", vocab)

    def test_all_corpora_roundtrip_and_reproduce_ratios(self) -> None:
        for language in self.stats["languages"]:
            with self.subTest(language=language["code"]):
                path = BASE_DIR / language["file"]
                text = path.read_text(encoding="utf-8")
                token_ids = self.submission.encode(text)
                decoded = self.submission.decode(token_ids)

                self.assertEqual(decoded, text)
                self.assertEqual(bpe.visible_text(decoded), bpe.visible_text(text))
                self.assertEqual(len(token_ids), language["scored_tokens"])
                self.assertEqual(bpe.faithful_units(text), language["faithful_units"])
                self.assertLessEqual(len(token_ids) / language["faithful_units"], 1.2)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    language["sha256"],
                )

    def test_unseen_unicode_and_assignment_sample_roundtrip(self) -> None:
        samples = (
            "India's population is 1,428,627,663.",
            "  leading\tspaces\nnew line  ",
            "भारत తెలుగు ಭಾರತ 🙂",
            "𓀀",
            "\n\t  ",
            "",
        )
        for text in samples:
            with self.subTest(text=text):
                token_ids = self.submission.encode(text)
                self.assertTrue(all(0 <= token_id < 10_000 for token_id in token_ids))
                self.assertEqual(self.submission.decode(token_ids), text)

    def test_artifact_works_directly_with_hugging_face_tokenizers(self) -> None:
        tokenizer = Tokenizer.from_file(str(BASE_DIR / "tokenizer.json"))
        text = "India's population is 1,428,627,663."
        encoding = tokenizer.encode(text)
        self.assertEqual(
            tokenizer.decode(encoding.ids, skip_special_tokens=False),
            text,
        )

    def test_checked_in_statistics_recompute(self) -> None:
        verified = build_tokenizer.verify(BASE_DIR)
        self.assertGreater(verified["spread"], 0.0)
        self.assertTrue(math.isfinite(verified["score"]))
        self.assertGreater(verified["score"], 80_000)
        self.assertTrue(verified["lossless"])
        self.assertTrue(verified["exact_roundtrip"])
        self.assertEqual(verified["unknown_tokens_emitted"], 0)
        self.assertEqual(verified["fourth_language"], "Kannada")


if __name__ == "__main__":
    unittest.main()
