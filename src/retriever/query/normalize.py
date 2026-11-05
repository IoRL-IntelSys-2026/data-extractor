"""C-RET-02 -- query normalisation.

UC-RET-01 step 2: trim, collapse whitespace runs, drop control characters and cap
the length (``query.max_tokens``).  Normalisation is deliberately conservative --
the query is already a standalone question by the time it reaches us, and heavy
rewriting would break the ``search_query`` field the caller relies on.
"""

from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str, *, max_tokens: int | None = None) -> str:
    """Return the cleaned query text.

    ``max_tokens`` caps the length in *whitespace tokens* (a cheap proxy that does
    not require the model tokenizer here); a query longer than the limit is
    truncated.  The tokenizer-aware cap happens later when embedding.
    """
    if text is None:
        return ""
    cleaned = _CONTROL_CHARS.sub(" ", text)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if max_tokens is not None:
        tokens = cleaned.split(" ")
        if len(tokens) > max_tokens:
            cleaned = " ".join(tokens[:max_tokens])
    return cleaned


def build_search_text(normalized: str, expansion: str | None) -> str:
    """Append the dictionary expansion to the normalised query.

    ``search_query`` is the text the search actually ran on: normalised query
    plus any expansion.  When the dictionary adds nothing, it is just the
    normalised query (K3 rules in readme 3.1.5).
    """
    if expansion:
        return f"{normalized} {expansion}".strip()
    return normalized


__all__ = ["normalize_query", "build_search_text"]
