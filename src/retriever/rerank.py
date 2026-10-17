"""C-RET-06 -- cross-encoder reranker (optional, after the first measurement).

Re-scores ``(search_text, embed_text)`` pairs with ``BAAI/bge-reranker-v2-m3``
using FlagEmbedding's ``FlagReranker`` (the same package D3 uses).  Scores are
normalised to 0..1.

When the reranker is enabled and fails, the request must fail with 503
(UC-RET-01) -- a silent fallback would corrupt both answers and evaluation, so
this module raises :class:`RerankerError` and the caller maps it to 503.
"""

from __future__ import annotations

from typing import Protocol

from retriever.config import RerankConfig
from retriever.search import Candidate


class RerankerError(RuntimeError):
    """Raised when the reranker is enabled but cannot produce scores."""


class Reranker(Protocol):
    def score(self, query: str, candidates: list[Candidate]) -> list[float]: ...


class _FlagReranker:
    """FlagEmbedding-backed cross-encoder."""

    def __init__(self, config: RerankConfig) -> None:
        try:
            from FlagEmbedding import FlagReranker  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RerankerError(
                "rerank.enabled=true but FlagEmbedding is not installed"
            ) from exc
        try:
            self._reranker = FlagReranker(config.model, use_fp16=True)
            self._config = config
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RerankerError(f"failed to load reranker {config.model!r}: {exc}") from exc

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, c.payload.get("embed_text", c.payload.get("text", ""))] for c in candidates]
        try:
            raw = self._reranker.compute_score(
                pairs, normalize=True, max_length=self._config.max_length
            )
        except TypeError:
            # Older/newer FlagEmbedding signatures may not accept max_length.
            raw = self._reranker.compute_score(pairs, normalize=True)
        except Exception as exc:  # pragma: no cover - runtime failure
            raise RerankerError(f"reranker failed: {exc}") from exc
        if isinstance(raw, (int, float)):
            raw = [float(raw)]
        return [float(x) for x in raw]


def build_reranker(config: RerankConfig) -> Reranker | None:
    """Return a reranker when enabled, else ``None``."""
    if not config.enabled:
        return None
    return _FlagReranker(config)


def rerank_candidates(reranker: Reranker, query: str, candidates: list[Candidate]) -> list[Candidate]:
    """Attach reranker scores to ``candidates`` and sort by them (descending).

    Fails loudly (``RerankerError``) rather than degrading silently.
    """
    if not candidates:
        return []
    try:
        scores = reranker.score(query, candidates)
    except RerankerError:
        raise
    except Exception as exc:
        raise RerankerError(f"reranker failed: {exc}") from exc
    if len(scores) != len(candidates):
        raise RerankerError(
            f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
        )
    for cand, score in zip(candidates, scores):
        cand.rerank_score = score
        cand.score = score
    return sorted(candidates, key=lambda c: c.score, reverse=True)


__all__ = ["Reranker", "RerankerError", "build_reranker", "rerank_candidates"]
