from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpe


BASE_DIR = Path(__file__).resolve().parent.parent / "assets"


class TokenizerArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = json.loads(
            (BASE_DIR / "tokenizer.json").read_text(encoding="utf-8")
        )
        cls.stats = json.loads((BASE_DIR / "stats.json").read_text(encoding="utf-8"))
        cls.submission = bpe.SubmissionTokenizer(cls.tokenizer)

    def test_global_vocabulary_has_exactly_ten_thousand_ids(self) -> None:
        tokens = self.tokenizer["tokens"]
        self.assertEqual(self.tokenizer["vocab_size"], 10_000)
        self.assertEqual(len(tokens), 10_000)
        self.assertEqual([token["id"] for token in tokens], list(range(10_000)))
        self.assertNotIn("profiles", self.tokenizer)
        self.assertEqual(self.tokenizer["encoding"]["type"], "protected-lexeme-bpe")

    def test_tokenizer_is_finite_lossless_and_below_limit(self) -> None:
        for language in self.stats["languages"]:
            with self.subTest(language=language["code"]):
                text = bpe.read_corpus(BASE_DIR / language["file"])
                words = bpe.split_words(text)
                scored_ids = self.submission.encode(text)
                self.assertEqual(len(scored_ids), language["scored_tokens"])
                self.assertLessEqual(len(scored_ids) / len(words), 1.2)

                for word in set(words):
                    token_ids = self.submission.encode_word(word)
                    self.assertEqual(self.submission.decode_word(token_ids), word)

    def test_unseen_word_uses_lossless_bpe_fallback(self) -> None:
        word = "unseen-token-🙂-987654321"
        self.assertNotIn(word, self.submission.lexemes)
        self.assertNotIn(word, self.submission.pieces)
        token_ids = self.submission.encode_word(word)
        self.assertEqual(self.submission.decode_word(token_ids), word)

    def test_checked_in_statistics_recompute(self) -> None:
        verified = bpe.verify(BASE_DIR)
        self.assertGreater(verified["spread"], 0.0)
        self.assertTrue(math.isfinite(verified["score"]))
        self.assertGreater(verified["score"], 600_000)
        self.assertTrue(verified["lossless"])
        self.assertEqual(verified["unknown_tokens_emitted"], 0)
        for language in verified["languages"]:
            self.assertTrue(language["source_url"].startswith("https://"))
            self.assertTrue(language["acquisition"])
            self.assertTrue(language["cleaning"])


if __name__ == "__main__":
    unittest.main()
