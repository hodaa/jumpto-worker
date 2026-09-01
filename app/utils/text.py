"""Text normalization helpers for the JumpTo worker."""

import re

_PUNCTUATION_RE = re.compile(r"[^\w']")


def normalize_word(word: str) -> str:
    """
    Normalize a word for matching: lowercase and strip punctuation.

    Args:
        word: Raw word string

    Returns:
        Normalized word
    """
    normalized = _PUNCTUATION_RE.sub("", word.lower())
    return normalized.strip("'")
