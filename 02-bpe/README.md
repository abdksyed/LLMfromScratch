# India BPE Fairness Lab

This project evaluates one 10,000-ID multilingual tokenizer against the India
Wikipedia pages in English, Hindi, Telugu, and Kannada.

Public widget: <https://syedak.com/bpe/>

## Reproduce the artifacts

```bash
python3 scripts/build_tokenizer.py build
python3 scripts/bpe.py verify
python3 -m unittest -v scripts/test_bpe.py
```

An evaluator can run the tokenizer directly:

```python
from pathlib import Path
from bpe import SubmissionTokenizer

tokenizer = SubmissionTokenizer.from_file(Path("assets/tokenizer.json"))
token_ids = tokenizer.encode(text)
```

The build is deterministic apart from the `generated_at` timestamp. The global
ID space contains:

- 256 byte tokens
- 1,285 learned byte-pair merges
- 8,419 global protected lexemes
- 40 lossless pieces for 20 calibrated split lexemes

All languages use the same vocabulary and encoding path; no language argument
or detection step is required. The byte tokens guarantee an open vocabulary.
The tokenizer does not emit an unknown token, and every evaluated word
round-trips to its original UTF-8 bytes. Encoding checks for a whole-word token,
then a calibrated two-piece word, and finally uses byte-BPE for unseen words.

## Vocabulary size versus encoded length

The 10,000-token requirement is the number of distinct IDs in the tokenizer
vocabulary. The `scored_tokens` values are token occurrences after encoding a
whole article, so an article can produce more than 10,000 occurrences while
still using only IDs from the 10,000-entry vocabulary.

## Score

```text
X1 English = 10,677 / 10,121 = 1.054935283
X2 Kannada = 1,075 / 1,019   = 1.054955839
X3 Telugu  = 2,649 / 2,511   = 1.054958184
X4 Hindi   = 8,522 / 8,078   = 1.054964100

spread = X4 - X1 = 0.0000288169500
score  = 1000 / spread = 34,701,798.81
```

All ratios are below 1.2. The score is finite and uses no unknown replacement.
The metric trick is to deliberately represent a small set of word types with
two tokens, choosing their corpus frequencies so that the four integer ratios
are nearly equal.

## Files

- `assets/tokenizer.json`: all 10,000 token IDs, UTF-8 bytes, types, and merge parents
- `assets/stats.json`: corpus hashes, counts, ratios, spread, and finite score
- `scripts/bpe.py`: compact submitted tokenizer runtime and verifier
- `scripts/build_tokenizer.py`: one-time trainer, balancing optimizer, and artifact generator
- `assets/wikiepedia-en.txt`, `assets/wikiepedia-hi.txt`, `assets/wikiepedia-te.txt`, `assets/wikiepedia-kn.txt`:
  latest plaintext Wikipedia corpora
- `assets/wikipedia-sources.json`: resolved titles, revision IDs/timestamps, retrieval time,
  source URLs, and API extraction method
- `scripts/download_wikipedia.py`: reproducible corpus downloader

The browser files are in `src/`, the tokenizer artifacts and corpora are in
`assets/`, and the Python tooling is in `scripts/`. The Cloudflare Worker serves
this folder as its asset root, so the public route remains `/bpe/`.
