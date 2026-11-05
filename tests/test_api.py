"""T-RET-01 at the HTTP boundary: /retrieve and /health behaviour.

We use ``TestClient`` *without* the context-manager form so the FastAPI lifespan
(which would try to connect to a live Qdrant) does not run.  Instead we populate
``api.state`` directly with a fake-backed retriever.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import retriever.api as api
from retriever.config import RetrieverConfig
from retriever.core import Retriever
from retriever.guard import CollectionStatus
from tests.conftest import FakeEmbedder, FakeQdrantClient


@pytest.fixture
def client():
    cfg = RetrieverConfig.model_validate(
        {
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
            "profile_name": "dense",
        }
    )
    retriever = Retriever(cfg, FakeQdrantClient(), embedder=FakeEmbedder())
    api.state.config = cfg
    api.state.retriever = retriever
    api.state.client = FakeQdrantClient()
    api.state.statuses = {
        "mdd_current": CollectionStatus("mdd_current", "mdd_chunks_v1", "BAAI/bge-m3", True)
    }
    api.state.reranker_loaded = False
    return TestClient(api.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["config_id"]
    assert body["embedding_model"] == "BAAI/bge-m3"
    assert body["collections"]["mdd_current"] == "mdd_chunks_v1"


def test_retrieve_ok(client):
    resp = client.post("/retrieve", json={"query": "shower in dorm"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["config_id"]
    assert body["search_query"]
    assert body["collection"] == "mdd_current"
    ranks = [c["rank"] for c in body["chunks"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_retrieve_history_rejected(client):
    assert client.post("/retrieve", json={"query": "hi", "history": []}).status_code == 422


def test_retrieve_empty_query_rejected(client):
    assert client.post("/retrieve", json={"query": ""}).status_code == 422


def test_retrieve_top_k_bounds(client):
    assert client.post("/retrieve", json={"query": "hello", "top_k": 0}).status_code == 422
    assert client.post("/retrieve", json={"query": "hello", "top_k": 999}).status_code == 422


def test_retrieve_collection_not_allowed(client):
    assert client.post("/retrieve", json={"query": "hello", "collection": "secret"}).status_code == 422


def test_retrieve_debug_flag(client):
    body = client.post("/retrieve", json={"query": "shower in dorm", "debug": True}).json()
    assert body["debug"] is not None
    body2 = client.post("/retrieve", json={"query": "shower in dorm", "debug": False}).json()
    assert body2.get("debug") is None


def test_unknown_section_returns_empty_no_relevant(client):
    body = client.post(
        "/retrieve",
        json={"query": "shower in dorm", "filters": {"section": ["nope"]}},
    ).json()
    assert body["chunks"] == []
    assert body["no_relevant"] is True
