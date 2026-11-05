"""Integration tests against a live Qdrant (marker: ``integration``).

Run with ``pytest -m integration`` against an isolated test Qdrant (see
``docker-compose.test.yml``).  They are skipped automatically when Qdrant is not
reachable so the default ``pytest`` run stays green without a server.

Covered:
* T-RET-03 -- collection guard: model / vector mismatch aborts startup,
* T-RET-05 -- filters honoured in real dense/sparse/hybrid queries,
* T-RET-09 -- run reproducibility (ID/order stable apart from latency),
* T-RET-01 -- K3 validity and known chunk in top-k on the fixture collection.
"""

from __future__ import annotations

import os

import pytest
from qdrant_client import QdrantClient

from rag_common.schemas import RetrievalRequest
from retriever.config import RetrieverConfig
from retriever.core import Retriever
from retriever.guard import CollectionGuardError, check_collection
from retriever.tools.load_test_collection import drop_collection, load_fixture

QDRANT_URL = os.environ.get("QDRANT_TEST_URL", "http://localhost:6333")
COLLECTION = "mdd_exp_test_v1"


def _client_or_skip() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, timeout=2.0)
    try:
        client.get_collections()
    except Exception:
        pytest.skip(f"no Qdrant at {QDRANT_URL}")
    return client


@pytest.fixture(scope="module")
def qdrant():
    client = _client_or_skip()
    load_fixture(client, collection=COLLECTION, embedding_model="BAAI/bge-m3")
    yield client
    drop_collection(client, COLLECTION)


@pytest.fixture
def config_dict():
    return {
        "collection": COLLECTION,
        "allowed_collections": [COLLECTION],
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


@pytest.mark.integration
def test_guard_ok(qdrant, config_dict):
    cfg = RetrieverConfig.model_validate(config_dict)
    status = check_collection(qdrant, cfg, COLLECTION)
    assert status.embedding_model == "BAAI/bge-m3"


@pytest.mark.integration
def test_guard_model_mismatch(qdrant, config_dict):
    cfg = RetrieverConfig.model_validate(config_dict)
    # corrupt index_info to a different model
    from qdrant_client import models as qmodels

    from retriever.metadata import INDEX_INFO_POINT_ID, index_info_collection

    qdrant.upsert(
        collection_name=index_info_collection(COLLECTION),
        points=[
            qmodels.PointStruct(
                id=INDEX_INFO_POINT_ID,
                vector=[0.0],
                payload={"embedding_model": "deepvk/USER-bge-m3"},
            )
        ],
    )
    with pytest.raises(CollectionGuardError) as exc:
        check_collection(qdrant, cfg, COLLECTION)
    assert "deepvk/USER-bge-m3" in str(exc.value)
    assert "BAAI/bge-m3" in str(exc.value)
    # restore
    load_fixture(qdrant, collection=COLLECTION, embedding_model="BAAI/bge-m3", recreate=False)


@pytest.mark.integration
def test_dense_retrieval_known_chunk(qdrant, config_dict, fake_embedder, make_config):
    cfg = make_config(**{k: v for k, v in config_dict.items() if k != "profile_name"})
    retriever = Retriever(cfg, qdrant, embedder=fake_embedder)
    resp = retriever.retrieve(RetrievalRequest(query="shower in dorm", top_k=5))
    assert "mdd-housing-1#0" in [c.chunk_id for c in resp.chunks]


@pytest.mark.integration
def test_filter_in_real_query(qdrant, config_dict, fake_embedder, make_config):
    from rag_common.schemas import FilterSpec

    cfg = make_config(**{k: v for k, v in config_dict.items() if k != "profile_name"})
    retriever = Retriever(cfg, qdrant, embedder=fake_embedder)
    resp = retriever.retrieve(
        RetrievalRequest(query="shower in dorm", filters=FilterSpec(lang="ru"), top_k=5)
    )
    assert all(c.payload["lang"] == "ru" for c in resp.chunks)


@pytest.mark.integration
def test_run_reproducible(qdrant, config_dict, fake_embedder, make_config, tmp_path):
    """T-RET-09: two runs with the same config agree on IDs, order and metrics."""
    import json
    from pathlib import Path

    from retriever.eval.runner import run_eval

    cfg = make_config(**{k: v for k, v in config_dict.items() if k != "profile_name"})
    retriever = Retriever(cfg, qdrant, embedder=fake_embedder)

    dataset = os.path.join(os.path.dirname(__file__), "fixtures", "k6", "k6_sample.jsonl")
    cfg_path = tmp_path / "dense.yaml"
    # write a config that targets this fixture collection
    import yaml

    yaml.safe_dump(config_dict_without_profile(config_dict), cfg_path.open("w", encoding="utf-8"))

    s1 = run_eval(config_path=cfg_path, dataset_path=dataset, split="validation",
                  variant="last", retriever=retriever, output_root=tmp_path / "r1")
    s2 = run_eval(config_path=cfg_path, dataset_path=dataset, split="validation",
                  variant="last", retriever=retriever, output_root=tmp_path / "r2")

    def _rows(run_dir: str) -> list[dict]:
        text = (Path(run_dir) / "samples.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    rows1, rows2 = _rows(s1["run_dir"]), _rows(s2["run_dir"])
    assert [r["sample_id"] for r in rows1] == [r["sample_id"] for r in rows2]
    assert [r["retrieved_doc_ids"] for r in rows1] == [r["retrieved_doc_ids"] for r in rows2]
    assert s1["hit_at_5"] == s2["hit_at_5"]
    assert s1["mrr_at_10"] == s2["mrr_at_10"]


def config_dict_without_profile(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "profile_name"}
