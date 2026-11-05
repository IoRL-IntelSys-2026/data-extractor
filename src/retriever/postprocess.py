"""C-RET-07 -- post-processing: max_per_doc, top_k, rank, no_relevant.

Implements INV-RET-08 (never exceed ``max_per_doc`` chunks of one ``doc_id``),
INV-RET-01 (rank 1..n without gaps, non-increasing score) and INV-RET-09 (the
``no_relevant`` rule: a single configured rule comparing the best candidate's
relevance signal against a threshold).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_common.schemas import RetrievedChunk
from retriever.config import NoRelevantConfig, PostprocessConfig
from retriever.search import Candidate


@dataclass
class PostprocessResult:
    chunks: list[RetrievedChunk]
    no_relevant: bool
    relevance_signal: float | None


def apply_max_per_doc(candidates: list[Candidate], max_per_doc: int | None) -> list[Candidate]:
    """Keep at most ``max_per_doc`` chunks of any single ``doc_id``.

    Order is preserved.  If the limit starves the result, we return fewer chunks
    rather than violate it (INV-RET-08).
    """
    if max_per_doc is None:
        return list(candidates)
    counts: dict[str, int] = {}
    kept: list[Candidate] = []
    for cand in candidates:
        doc_id = str(cand.payload.get("doc_id", ""))
        if counts.get(doc_id, 0) >= max_per_doc:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        kept.append(cand)
    return kept


def _relevance_signal(candidates: list[Candidate], no_relevant: NoRelevantConfig) -> float | None:
    """Best candidate's signal according to the configured ``signal`` kind."""
    if not candidates:
        return None
    if no_relevant.signal == "rerank_score":
        best = max((c.rerank_score if c.rerank_score is not None else c.score) for c in candidates)
        return float(best)
    # cosine / rrf / dot all come straight from the candidate score.
    return float(max(c.score for c in candidates))


def compute_no_relevant(
    candidates: list[Candidate],
    no_relevant: NoRelevantConfig,
) -> tuple[bool, float | None]:
    """Apply the single ``no_relevant`` rule (INV-RET-09).

    * 0 candidates -> ``True`` regardless of threshold,
    * ``threshold is None`` -> the rule is off; only the empty case triggers,
    * otherwise -> ``True`` when the best signal is below the threshold.
    """
    if not candidates:
        return True, None
    signal = _relevance_signal(candidates, no_relevant)
    if no_relevant.threshold is None:
        return False, signal
    return (signal is not None and signal < no_relevant.threshold), signal


def to_chunk(candidate: Candidate, rank: int) -> RetrievedChunk:
    payload = candidate.payload
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id", candidate.chunk_id)),
        doc_id=str(payload.get("doc_id", "")),
        url=str(payload.get("url", "")),
        title=str(payload.get("title", "")),
        text=str(payload.get("text", "")),
        score=float(candidate.score),
        rank=rank,
        payload=payload or None,
    )


def postprocess(
    candidates: list[Candidate],
    *,
    postprocess_config: PostprocessConfig,
    no_relevant_config: NoRelevantConfig,
    top_k: int,
) -> PostprocessResult:
    """Full post-processing pipeline for one query."""
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    no_rel, signal = compute_no_relevant(ordered, no_relevant_config)
    limited = apply_max_per_doc(ordered, postprocess_config.max_per_doc)
    selected = limited[:top_k]
    chunks = [to_chunk(cand, rank=i) for i, cand in enumerate(selected, start=1)]
    return PostprocessResult(chunks=chunks, no_relevant=no_rel, relevance_signal=signal)


__all__ = [
    "postprocess",
    "PostprocessResult",
    "apply_max_per_doc",
    "compute_no_relevant",
    "to_chunk",
]
