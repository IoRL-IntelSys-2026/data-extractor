"""C-RET-05 -- Qdrant search (dense / sparse / hybrid).

Invariants enforced here:

* only ``query_points`` is used -- ``search()`` and ``recommend()`` were removed in
  Qdrant 1.19 (INV-RET-02),
* the RRF ``k`` comes from the config and is always explicit,
* the filter is passed to *every* prefetch branch as well as the top level
  (INV-RET-06) -- whether the top-level ``query_filter`` propagates to prefetch
  depends on the server version, so we do not rely on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from rag_common.embeddings import Embedder
from retriever.config import RetrieverConfig

FilterMap = dict[str, list[str]]


@dataclass
class Candidate:
    """A single search hit before post-processing."""

    chunk_id: str
    score: float
    payload: dict
    rank: int = 0
    fusion_rank: int | None = None
    rerank_score: float | None = None
    extra: dict = field(default_factory=dict)


def build_filter(filters: FilterMap | None) -> models.Filter | None:
    """Translate a K3 filter map into a Qdrant ``Filter``.

    Single values use ``MatchValue``, lists use ``MatchAny`` ("any of").
    """
    if not filters:
        return None
    must: list[models.FieldCondition] = []
    for key, values in filters.items():
        if not values:
            continue
        if len(values) == 1:
            cond = models.FieldCondition(key=key, match=models.MatchValue(value=values[0]))
        else:
            cond = models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))
        must.append(cond)
    return models.Filter(must=must) if must else None


def _point_to_candidate(point: models.ScoredPoint) -> Candidate:
    payload = dict(point.payload or {})
    chunk_id = payload.get("chunk_id") or str(point.id)
    return Candidate(chunk_id=chunk_id, score=float(point.score), payload=payload)


class QdrantSearcher:
    """Executes the configured search mode against a Qdrant collection."""

    def __init__(self, client: QdrantClient, config: RetrieverConfig, embedder: Embedder) -> None:
        self.client = client
        self.config = config
        self.embedder = embedder

    def search(
        self,
        collection: str,
        search_text: str,
        filters: FilterMap | None,
    ) -> list[Candidate]:
        """Run the search and return the top ``candidates`` hits."""
        mode = self.config.search.mode
        if mode == "dense":
            return self._dense(collection, search_text, filters)
        if mode == "sparse":
            return self._sparse(collection, search_text, filters)
        if mode == "hybrid":
            return self._hybrid(collection, search_text, filters)
        raise ValueError(f"unsupported search mode: {mode}")  # pragma: no cover

    # -- modes ---------------------------------------------------------------

    def _dense(self, collection: str, text: str, filters: FilterMap | None) -> list[Candidate]:
        vector = self.embedder.encode_query(text, with_sparse=False).dense
        res = self.client.query_points(
            collection_name=collection,
            query=_to_list(vector),
            using="dense",
            query_filter=build_filter(filters),
            limit=self.config.search.candidates,
            with_payload=True,
        )
        return [_point_to_candidate(p) for p in res.points]

    def _sparse(self, collection: str, text: str, filters: FilterMap | None) -> list[Candidate]:
        q = self.embedder.encode_query(text, with_sparse=True)
        if q.sparse is None:
            raise RuntimeError(
                f"embedding model {self.embedder.spec.name!r} provides no sparse vectors "
                "but mode=sparse was requested"
            )
        res = self.client.query_points(
            collection_name=collection,
            query=models.SparseVector(indices=q.sparse.indices, values=q.sparse.values),
            using="sparse",
            query_filter=build_filter(filters),
            limit=self.config.search.candidates,
            with_payload=True,
        )
        return [_point_to_candidate(p) for p in res.points]

    def _hybrid(self, collection: str, text: str, filters: FilterMap | None) -> list[Candidate]:
        q = self.embedder.encode_query(text, with_sparse=True)
        if q.sparse is None:
            raise RuntimeError(
                f"embedding model {self.embedder.spec.name!r} provides no sparse vectors "
                "but mode=hybrid was requested"
            )
        flt = build_filter(filters)
        fusion = self.config.search.fusion
        assert fusion is not None  # validated in config
        prefetch = [
            models.Prefetch(query=_to_list(q.dense), using="dense", limit=self.config.search.prefetch_limit, filter=flt),
            models.Prefetch(
                query=models.SparseVector(indices=q.sparse.indices, values=q.sparse.values),
                using="sparse",
                limit=self.config.search.prefetch_limit,
                filter=flt,
            ),
        ]
        res = self.client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.RrfQuery(rrf=models.Rrf(k=fusion.k)),
            query_filter=flt,
            limit=self.config.search.candidates,
            with_payload=True,
        )
        candidates = [_point_to_candidate(p) for p in res.points]
        for i, cand in enumerate(candidates):
            cand.fusion_rank = i
        return candidates


def _to_list(vector) -> list[float]:
    """Normalise a numpy vector to a plain list for the Qdrant client."""
    if vector is None:
        return []
    if isinstance(vector, list):
        return vector
    return [float(x) for x in vector]


__all__ = ["QdrantSearcher", "Candidate", "build_filter", "FilterMap"]
