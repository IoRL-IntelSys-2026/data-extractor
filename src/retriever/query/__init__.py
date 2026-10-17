"""Query processing: normalisation and dictionary expansion (C-RET-02)."""

from retriever.query.dictionary import Dictionary
from retriever.query.normalize import build_search_text, normalize_query

__all__ = ["Dictionary", "normalize_query", "build_search_text"]
