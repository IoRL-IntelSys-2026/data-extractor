"""K3 schema rules: history -> 422, empty query -> 422, extras forbidden (INV-RET-05)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_common.schemas import (
    RetrievedChunk,
    RetrievalRequest,
    RetrievalResponse,
)


def test_history_field_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="hello", history=[{"role": "user", "content": "x"}])


def test_empty_query_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="")
    with pytest.raises(ValidationError):
        RetrievalRequest(query="   ")


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="hello", unexpected=1)


def test_response_search_query_not_null():
    with pytest.raises(ValidationError):
        RetrievalResponse(
            search_query=None,
            no_relevant=False,
            collection="c",
            config_id="p@12345678",
            latency_ms=1.0,
        )


def test_response_accepts_chunks():
    resp = RetrievalResponse(
        search_query="q",
        no_relevant=False,
        collection="c",
        config_id="p@12345678",
        latency_ms=1.0,
        chunks=[
            RetrievedChunk(
                chunk_id="a#0", doc_id="a", url="u", title="t", text="x", score=1.0, rank=1
            )
        ],
    )
    assert resp.chunks[0].rank == 1
