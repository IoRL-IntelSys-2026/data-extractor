"""Query embedding module (D3) -- the retriever's only source of vectors.

The readme is explicit (INV-RET-03): the query embedding is *always* computed
through ``rag_common.embeddings`` using a model from its registry, and the model
name used by the retriever must match the one the collection was built with.

Two things live here:

* :class:`ModelRegistry` -- maps a model id to a small descriptor (dimension,
  dense-only or dense+sparse, query/document prefixes, max tokens),
* :func:`get_embedder` -- returns an object exposing ``encode_query(text)``.

For the heavyweight ``bge-m3`` model the real implementation uses
``FlagEmbedding``; to keep the retriever installable and testable without pulling
torch, the concrete model is loaded lazily and can be replaced with a fake
embedder in tests (:func:`set_embedder`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class QueryVector:
    """The result of encoding a query: a dense vector and, optionally, a sparse one."""

    dense: np.ndarray | None = None
    sparse: "SparseVec | None" = None


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class ModelSpec:
    """Descriptor for an embedding model in the registry (D3)."""

    name: str
    dimension: int
    supports_sparse: bool = False
    # Prefixes applied to queries / documents per the model's card.  FRIDA and
    # e5 degrade without them (see readme, section 3.2.3).
    query_prefix: str = ""
    document_prefix: str = ""
    max_tokens: int = 512


# Registry of the models mentioned across the readme.  Real embedding backends
# are attached lazily in :func:`get_embedder`.
REGISTRY: dict[str, ModelSpec] = {
    "BAAI/bge-m3": ModelSpec("BAAI/bge-m3", dimension=1024, supports_sparse=True, max_tokens=8192),
    "deepvk/USER-bge-m3": ModelSpec("deepvk/USER-bge-m3", dimension=1024, supports_sparse=False),
    "ai-forever/FRIDA": ModelSpec(
        "ai-forever/FRIDA",
        dimension=1536,
        supports_sparse=False,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        max_tokens=512,
    ),
    "intfloat/multilingual-e5-large": ModelSpec(
        "intfloat/multilingual-e5-large",
        dimension=1024,
        supports_sparse=False,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_tokens=512,
    ),
}


class Embedder(Protocol):
    """Minimal embedder interface the retriever depends on (D3)."""

    spec: ModelSpec

    def encode_query(self, text: str, *, with_sparse: bool = True) -> QueryVector: ...


def spec_for(model_name: str) -> ModelSpec:
    """Look up a model in the registry, raising a clear error when unknown."""
    try:
        return REGISTRY[model_name]
    except KeyError as exc:  # pragma: no cover - defensive
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown embedding model {model_name!r}; known models: {known}") from exc


class _FlagEmbedder:
    """Real embedder backed by FlagEmbedding's ``BGEM3FlagModel``.

    Imported lazily so that importing :mod:`rag_common.embeddings` does not pull
    in torch.
    """

    def __init__(self, model_name: str, *, use_fp16: bool = True) -> None:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

        self.spec = spec_for(model_name)
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode_query(self, text: str, *, with_sparse: bool = True) -> QueryVector:
        out = self._model.encode(
            [self.spec.query_prefix + text],
            return_dense=True,
            return_sparse=with_sparse and self.spec.supports_sparse,
        )
        dense = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        sparse = None
        if with_sparse and self.spec.supports_sparse:
            weights = out.get("lexical_weights", [{}])[0]
            indices = [int(k) for k in weights]
            values = [float(v) for v in weights.values()]
            sparse = SparseVec(indices=indices, values=values)
        return QueryVector(dense=dense, sparse=sparse)


_embedder: Embedder | None = None
_embedder_factory: Callable[[str], Embedder] = lambda name: _FlagEmbedder(name)


def set_embedder_factory(factory: Callable[[str], Embedder]) -> None:
    """Override how embedders are constructed (used by tests to inject fakes)."""
    global _embedder_factory, _embedder
    _embedder_factory = factory
    _embedder = None


def set_embedder(embedder: Embedder | None) -> None:
    """Install a ready-made embedder instance (tests) or reset with ``None``."""
    global _embedder
    _embedder = embedder


def get_embedder(model_name: str = "BAAI/bge-m3") -> Embedder:
    """Return the process-wide embedder, constructing it on first use."""
    global _embedder
    if _embedder is None or _embedder.spec.name != model_name:
        _embedder = _embedder_factory(model_name)
    return _embedder


__all__ = [
    "Embedder",
    "ModelSpec",
    "QueryVector",
    "SparseVec",
    "REGISTRY",
    "get_embedder",
    "spec_for",
    "set_embedder",
    "set_embedder_factory",
]
