"""TSK-RET-11 -- deterministic typo generator for hard-query sets.

Two edit kinds: drop a letter, or swap two adjacent letters.  Numbers and 1-2
letter words are left untouched.  Same ``seed`` gives the same output.
"""

from __future__ import annotations

import random


def _editable(word: str) -> bool:
    core = word.strip(".,!?;:()\"'")
    if len(core) < 3:
        return False
    return not any(ch.isdigit() for ch in core)


def make_typos(text: str, n: int = 1, seed: int = 0) -> str:
    """Introduce up to ``n`` typos into ``text``.

    Only words of 3+ letters (and containing no digits) are edited.  ``n`` caps
    the number of *base* edits applied; each edit is either a deletion or a
    transposition.
    """
    rng = random.Random(seed)
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if _editable(w)]
    if not candidates or n <= 0:
        return text

    rng.shuffle(candidates)
    chosen = candidates[:n]
    for idx in chosen:
        word = words[idx]
        if len(word) < 3:
            continue
        if rng.random() < 0.5:
            # drop a letter
            pos = rng.randrange(len(word))
            words[idx] = word[:pos] + word[pos + 1 :]
        else:
            # swap two adjacent letters
            pos = rng.randrange(len(word) - 1)
            words[idx] = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
    return " ".join(words)


__all__ = ["make_typos"]
