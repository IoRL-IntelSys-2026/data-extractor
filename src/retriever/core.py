"""C-RET-08 -- the orchestrator.

``Retriever.retrieve(RetrievalRequest) -> RetrievalResponse`` wires the pipeline in
the order of UC-RET-01:

normalise -> dictionary -> embed (D3, dense+sparse) -> search (dense/sparse/hybrid)
-> rerank (optional) -> max_per_doc -> top_k + rank + no_relevant -> K3 response.

The class never calls an LLM (INV-RET-05) and only talks to Qdrant through
:class:`~retriever.search.QdrantSearcher` (INV-RET-02).
"""

from __future__ import annotations

import time
from pathlib import Path

from qdrant_client import QdrantClient

from rag_common.embeddings import get_embedder
from rag_common.schemas import DebugInfo, RetrievalRequest, RetrievalResponse
from retriever.config import RetrieverConfig
from retriever.postprocess import postprocess
from retriever.query.dictionary import load_dictionary
from retriever.query.normalize import build_search_text, normalize_query
from retriever.rerank import RerankerError, build_reranker, rerank_candidates
from retriever.search import QdrantSearcher


class Retriever:
    """Stateless retriever bound to a config, a Qdrant client and an embedder."""

    def __init__(
        self,
        config: RetrieverConfig,
        client: QdrantClient,
        *,
        embedder=None,
        reranker=None,
    ) -> None:
        self.config = config
        self.client = client
        self.embedder = embedder or get_embedder(config.embedding.model)
        self._searcher = QdrantSearcher(client, config, self.embedder)

        # The dictionary is optional and loadable without a config base dir.
        dict_cfg = config.query.dictionary
        self.dictionary = None
        if dict_cfg.enabled:
            path = Path(dict_cfg.path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            self.dictionary = load_dictionary(path)

        self.reranker = reranker if reranker is not None else build_reranker(config.rerank)

    # -- public API ----------------------------------------------------------

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        started = time.perf_counter()
        timing: dict[str, float] = {}

        # 1. normalise (UC-RET-01 step 2)
        t0 = time.perf_counter()
        normalized = normalize_query(request.query, max_tokens=self.config.query.max_tokens)
        if not normalized:
            # Should already be caught by schema validation; defensive.
            normalized = request.query.strip()
        timing["normalize_ms"] = (time.perf_counter() - t0) * 1000

        # 2. dictionary expansion -> search_query (step 4)
        expansion = self.dictionary.expand(normalized) if self.dictionary else ""
        search_text = build_search_text(normalized, expansion)

        # 3. resolve collection (K3 allows overriding within allowed_collections)
        collection = request.collection or self.config.collection

        # 4. search (steps 5-7)
        t0 = time.perf_counter()
        candidates = self._searcher.search(collection, search_text, _filters(request))
        timing["search_ms"] = (time.perf_counter() - t0) * 1000

        # 5. rerank (step 8) -- failure is fatal (503), never silent
        if self.config.rerank.enabled and self.reranker is not None:
            t0 = time.perf_counter()
            try:
                candidates = rerank_candidates(self.reranker, search_text, candidates)
            except RerankerError:
                raise
            timing["rerank_ms"] = (time.perf_counter() - t0) * 1000

        # 6. postprocess (steps 9-10)
        top_k = request.top_k or self.config.top_k_default
        result = postprocess(
            candidates,
            postprocess_config=self.config.postprocess,
            no_relevant_config=self.config.no_relevant,
            top_k=top_k,
        )

        latency_ms = (time.perf_counter() - started) * 1000

        debug = None
        if request.debug:
            debug = DebugInfo(
                timing_ms=timing,
                candidates=[
                    {
                        "chunk_id": c.chunk_id,
                        "score": c.score,
                        "fusion_rank": c.fusion_rank,
                        "rerank_score": c.rerank_score,
                    }
                    for c in candidates[: self.config.search.candidates]
                ],
                relevance_signal=result.relevance_signal,
                resolved_collection=collection,
            )

        return RetrievalResponse(
            search_query=search_text,
            no_relevant=result.no_relevant,
            collection=collection,
            config_id=self.config.config_id,
            latency_ms=round(latency_ms, 3),
            chunks=result.chunks,
            debug=debug,
        )


def _filters(request: RetrievalRequest) -> dict[str, list[str]] | None:
    if request.filters is None:
        return None
    filters = request.filters.as_map()
    return filters or None


__all__ = ["Retriever"]
