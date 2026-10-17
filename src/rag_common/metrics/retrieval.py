"""Document-level retrieval metrics (D6).

All metrics work on *document ids* (the readme: "все метрики считаются на уровне
документов").  On mdd every turn has exactly one reference document, so Hit@5 and
Recall@5 coincide; on RUDN there may be several reference documents.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _as_list(values: Iterable[str] | None) -> list[str]:
    return list(values) if values is not None else []


def hit_at_k(
    retrieved_doc_ids: Sequence[str],
    reference_doc_ids: Iterable[str],
    k: int = 5,
) -> float:
    """1.0 if any reference document is within the first ``k`` retrieved ones."""
    refs = set(_as_list(reference_doc_ids))
    if not refs:
        return 0.0
    return 1.0 if refs & set(retrieved_doc_ids[:k]) else 0.0


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    reference_doc_ids: Iterable[str],
    k: int = 5,
) -> float:
    """Fraction of reference documents found within the first ``k`` retrieved."""
    refs = set(_as_list(reference_doc_ids))
    if not refs:
        return 0.0
    found = sum(1 for doc in retrieved_doc_ids[:k] if doc in refs)
    return found / len(refs)


def mrr_at_k(
    retrieved_doc_ids: Sequence[str],
    reference_doc_ids: Iterable[str],
    k: int = 10,
) -> float:
    """Reciprocal rank of the first relevant document, 0.0 if none within ``k``."""
    refs = set(_as_list(reference_doc_ids))
    if not refs:
        return 0.0
    for rank, doc in enumerate(retrieved_doc_ids[:k], start=1):
        if doc in refs:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: Sequence[str],
    gains: dict[str, float],
    k: int = 5,
) -> float:
    """nDCG@k from graded relevance ``gains`` (0/1/2 manual labels).

    ``gains`` maps a document id to its graded relevance.  Only documents present
    in ``gains`` contribute; the ideal ranking uses the top gains overall.
    """
    import math

    def dcg(ranking: Sequence[str]) -> float:
        total = 0.0
        for i, doc in enumerate(ranking[:k], start=1):
            gain = gains.get(doc, 0.0)
            if gain:
                total += gain / math.log2(i + 1)
        return total

    ideal_order = sorted(gains.values(), reverse=True)
    idcg = 0.0
    for i, gain in enumerate(ideal_order[:k], start=1):
        if gain:
            idcg += gain / math.log2(i + 1)
    if idcg == 0.0:
        return 0.0
    return dcg(retrieved_doc_ids) / idcg


__all__ = ["hit_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k"]
