"""Shared test fixtures.

We do not download bge-m3 in CI.  Instead:

* :class:`FakeEmbedder` returns the stored query vector for a known question
  (from the K2 fixture) and a deterministic vector otherwise,
* :func:`fake_retriever` builds a :class:`Retriever` wired to an in-memory
  Qdrant-like client that ranks fixture points by cosine similarity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rag_common.embeddings import ModelSpec, QueryVector, SparseVec, set_embedder
from retriever.config import RetrieverConfig

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "k2"


def _load_points() -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURE_DIR / "points.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_queries() -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURE_DIR / "query_vectors.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


class FakeEmbedder:
    """Embedder that looks up ready-made vectors by question text."""

    spec = ModelSpec("BAAI/bge-m3", dimension=1024, supports_sparse=True)

    def __init__(self) -> None:
        self._by_text = {q["query"]: q for q in _load_queries()}

    def encode_query(self, text: str, *, with_sparse: bool = True) -> QueryVector:
        entry = self._by_text.get(text)
        if entry is None:
            dense = [0.0] * 1024
        else:
            dense = entry["dense"]
        sparse = None
        if with_sparse and entry is not None:
            sparse = SparseVec(indices=entry["sparse"]["indices"], values=entry["sparse"]["values"])
        return QueryVector(dense=dense, sparse=sparse)


class FakePoint:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    """In-memory stand-in for ``QdrantClient.query_points`` used by unit tests."""

    def __init__(self, points: list[dict] | None = None) -> None:
        self.points = points if points is not None else _load_points()
        # signature spy for T-RET-02
        self.calls: list[dict] = []

    def query_points(
        self,
        *,
        collection_name,
        query=None,
        using=None,
        query_filter=None,
        limit=10,
        with_payload=True,
        prefetch=None,
    ):
        self.calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "using": using,
                "limit": limit,
                "prefetch": prefetch,
                "query_filter": query_filter,
            }
        )
        if prefetch is not None:
            # hybrid: rank by fused position using the dense branch only (enough
            # for the unit test to verify call shape)
            dense_prefetch = prefetch[0]
            qvec = dense_prefetch.query
            scored = self._rank(qvec)
        else:
            qvec = query
            if hasattr(qvec, "indices"):  # sparse query -> use random-ish order
                scored = self._rank_sparse(qvec)
            else:
                scored = self._rank(qvec)
        scored = _apply_filter(scored, query_filter)
        return FakeQueryResult(scored[:limit])

    def _rank(self, qvec):
        out = []
        for p in self.points:
            score = _cosine(list(qvec), p["dense"])
            out.append(FakePoint(p["id"], score, p["payload"]))
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _rank_sparse(self, qvec):
        # deterministic pseudo-score from the query indices
        out = []
        for p in self.points:
            overlap = len(set(qvec.indices) & set(p["sparse"]["indices"]))
            out.append(FakePoint(p["id"], float(overlap), p["payload"]))
        out.sort(key=lambda x: x.score, reverse=True)
        return out


def _apply_filter(points: list[FakePoint], flt) -> list[FakePoint]:
    if flt is None:
        return points
    kept = []
    for p in points:
        ok = True
        for cond in flt.must or []:
            value = p.payload.get(cond.key)
            match = cond.match
            if hasattr(match, "value"):
                if value != match.value:
                    ok = False
                    break
            elif hasattr(match, "any"):
                if value not in match.any:
                    ok = False
                    break
        if ok:
            kept.append(p)
    return kept


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    embedder = FakeEmbedder()
    set_embedder(embedder)
    yield embedder
    set_embedder(None)


@pytest.fixture
def base_config_dict() -> dict:
    return {
        "collection": "mdd_current",
        "allowed_collections": ["mdd_current", "rudn_current"],
        "embedding": {"model": "BAAI/bge-m3"},
        "query": {"max_tokens": 256, "dictionary": {"enabled": False}},
        "search": {"mode": "dense", "prefetch_limit": 50, "candidates": 40},
        "rerank": {"enabled": False},
        "postprocess": {"max_per_doc": 2},
        "top_k_default": 5,
        "top_k_max": 20,
        "no_relevant": {"signal": "cosine", "threshold": None},
    }


@pytest.fixture
def make_config(base_config_dict):
    def _make(**overrides) -> RetrieverConfig:
        data = json.loads(json.dumps(base_config_dict))
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
        data.setdefault("profile_name", "dense")
        return RetrieverConfig.model_validate(data)

    return _make


@pytest.fixture
def point_ids_by_chunk():
    return {p["payload"]["chunk_id"]: p["id"] for p in _load_points()}
