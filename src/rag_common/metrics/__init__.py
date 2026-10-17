"""Retrieval metrics (D6) -- computed at the document level, without an LLM."""

from rag_common.metrics.retrieval import hit_at_k, mrr_at_k, recall_at_k, ndcg_at_k

__all__ = ["hit_at_k", "mrr_at_k", "recall_at_k", "ndcg_at_k"]
