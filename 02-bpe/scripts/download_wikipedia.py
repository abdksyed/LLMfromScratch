"""Backward-compatible entry point for the faithful corpus builder.

The old assignment prototype downloaded clipped plaintext extracts. The
submission now uses the full REST HTML conversion so links, tables, references,
and visible punctuation remain in the scored corpus.
"""

from build_faithful_corpus import main


if __name__ == "__main__":
    main()
