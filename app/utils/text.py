"""Text normalization helpers for the JumpTo worker."""

import re

_PUNCTUATION_RE = re.compile(r"[^\w']")

_ARABIC_SCRIPT_BLOCKS = (range(0x0600, 0x0700), range(0x0750, 0x0780))


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


def detect_language_from_title(title: str) -> str:
    """
    Infer a speech-to-text language tag from a video title's script.

    Titles containing any character in an Arabic-script Unicode block are
    treated as Arabic; everything else defaults to English.

    Args:
        title: Video title

    Returns:
        Language tag ("ar" or "en")
    """
    if any(ord(char) in block for block in _ARABIC_SCRIPT_BLOCKS for char in title):
        return "ar"
    return "en"
