"""T-RET-01 (K3 + known chunk in top-k) and T-RET-05 (filters always honoured)."""

from __future__ import annotations

import pytest

from rag_common.schemas import FilterSpec
from retriever.core import Retriever
from retriever.search import build_filter
from tests.conftest import FakeQdrantClient, _load_queries


def _make_retriever(make_config, fake_embedder, **overrides) -> Retriever:
    cfg = make_config(**overrides)
    return Retriever(cfg, FakeQdrantClient(), embedder=fake_embedder)


# -- T-RET-01 ---------------------------------------------------------------

def test_known_chunk_in_topk(make_config, fake_embedder):
    retriever = _make_retriever(make_config, fake_embedder)
    for q in _load_queries():
        from rag_common.schemas import RetrievalRequest

        resp = retriever.retrieve(RetrievalRequest(query=q["query"], top_k=5))
        assert q["target_chunk_id"] in [c.chunk_id for c in resp.chunks], q["query"]


def test_rank_and_score_monotonic(make_config, fake_embedder):
    from rag_common.schemas import RetrievalRequest

    retriever = _make_retriever(make_config, fake_embedder)
    resp = retriever.retrieve(RetrievalRequest(query="shower in dorm", top_k=5))
    ranks = [c.rank for c in resp.chunks]
    assert ranks == list(range(1, len(ranks) + 1))
    scores = [c.score for c in resp.chunks]
    assert scores == sorted(scores, reverse=True)
    for c in resp.chunks:
        assert c.chunk_id and c.doc_id and c.url and c.title and c.text


def test_chunks_capped_at_top_k(make_config, fake_embedder):
    from rag_common.schemas import RetrievalRequest

    retriever = _make_retriever(make_config, fake_embedder)
    resp = retriever.retrieve(RetrievalRequest(query="driver license renewal", top_k=2))
    assert len(resp.chunks) <= 2


# -- T-RET-05 ---------------------------------------------------------------

def test_section_any_of(make_config, fake_embedder):
    from rag_common.schemas import RetrievalRequest

    retriever = _make_retriever(make_config, fake_embedder)
    resp = retriever.retrieve(
        RetrievalRequest(
            query="driver license renewal",
            filters=FilterSpec(section=["dmv"]),
        )
    )
    assert resp.chunks
    assert all(c.payload["section"] == "dmv" for c in resp.chunks)


def test_lang_filter(make_config, fake_embedder):
    from rag_common.schemas import RetrievalRequest

    retriever = _make_retriever(make_config, fake_embedder)
    resp = retriever.retrieve(
        RetrievalRequest(query="щ", filters=FilterSpec(lang="ru"))
    )
    assert all(c.payload["lang"] == "ru" for c in resp.chunks)


def test_unknown_filter_value_empty_and_no_relevant(make_config, fake_embedder):
    from rag_common.schemas import RetrievalRequest

    retriever = _make_retriever(make_config, fake_embedder)
    resp = retriever.retrieve(
        RetrievalRequest(query="driver license renewal", filters=FilterSpec(section=["nope"]))
    )
    assert resp.chunks == []
    assert resp.no_relevant is True


def test_unknown_filter_field_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FilterSpec(**{"year": "2024"})


def test_build_filter_single_vs_list():
    flt = build_filter({"lang": ["ru"], "section": ["a", "b"]})
    kinds = {c.key: type(c.match).__name__ for c in flt.must}
    assert kinds["lang"] == "MatchValue"
    assert kinds["section"] == "MatchAny"
