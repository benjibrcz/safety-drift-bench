"""Hard-constraint scoring for DriftBench.

Default policy: **strict substring** — any occurrence of the forbidden word
(case-insensitive) counts as a failure, including inside compound words
like "pineapple".  This is the conservative choice and is documented/tested.
"""

from __future__ import annotations


def contains_forbidden_word(
    text: str,
    word: str = "apple",
    strict_substring: bool = True,
) -> bool:
    """Return True if *text* contains the forbidden *word*.

    Parameters
    ----------
    text : str
        The assistant response to check.
    word : str
        The forbidden word (default ``"apple"``).
    strict_substring : bool
        If True (default), any substring match counts (e.g. "pineapple"
        triggers on "apple").  If False, only whole-word matches count.
    """
    text_lower = text.casefold()
    word_lower = word.casefold()

    if strict_substring:
        return word_lower in text_lower

    # Whole-word matching via simple boundary check
    import re
    return bool(re.search(rf"\b{re.escape(word_lower)}\b", text_lower))
