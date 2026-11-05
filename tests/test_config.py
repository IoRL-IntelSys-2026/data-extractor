"""T-RET-04 (config_id) and T-RET-02 (explicit RRF k, only query_points)."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from retriever.config import RetrieverConfig
from retriever.search import QdrantSearcher


# -- T-RET-04 ---------------------------------------------------------------

def test_config_id_differs_when_threshold_differs(base_config_dict):
    a = copy.deepcopy(base_config_dict)
    a["no_relevant"] = {"signal": "rerank_score", "threshold": 0.15}
    b = copy.deepcopy(base_config_dict)
    b["no_relevant"] = {"signal": "rerank_score", "threshold": 0.20}
    ca = RetrieverConfig.model_validate({**a, "profile_name": "p"})
    cb = RetrieverConfig.model_validate({**b, "profile_name": "p"})
    assert ca.config_id != cb.config_id


def test_config_id_ignores_key_order_and_comments(base_config_dict):
    a = RetrieverConfig.model_validate({**base_config_dict, "profile_name": "p"})
    # reordering keys of the dict must not change the hash
    reordered = dict(reversed(list(base_config_dict.items())))
    b = RetrieverConfig.model_validate({**reordered, "profile_name": "p"})
    assert a.config_id == b.config_id


def test_config_id_format(base_config_dict):
    cfg = RetrieverConfig.model_validate({**base_config_dict, "profile_name": "hybrid_rrf_rerank"})
    name, _, digest = cfg.config_id.partition("@")
    assert name == "hybrid_rrf_rerank"
    assert len(digest) == 8
    int(digest, 16)  # valid hex


# -- T-RET-02 ---------------------------------------------------------------

def test_hybrid_requires_explicit_rrf_k(base_config_dict):
    data = copy.deepcopy(base_config_dict)
    data["search"] = {"mode": "hybrid", "prefetch_limit": 50, "candidates": 40}
    with pytest.raises(ValidationError):
        RetrieverConfig.model_validate({**data, "profile_name": "p"})


def test_hybrid_uses_config_rrf_k(fake_embedder, make_config):
    cfg = make_config(
        search={"mode": "hybrid", "prefetch_limit": 50, "candidates": 40, "fusion": {"type": "rrf", "k": 60}}
    )
    client = _FakeClientRecording()
    searcher = QdrantSearcher(client, cfg, fake_embedder)
    searcher.search("mdd_current", "shower in dorm", None)
    assert client.last_call["prefetch"] is not None
    rrf = client.last_call["query"]
    assert rrf.rrf.k == 60


class _FakeClientRecording:
    def __init__(self):
        self.last_call = {}

    def query_points(self, **kwargs):
        self.last_call = kwargs

        class _R:
            points = []

        return _R()
