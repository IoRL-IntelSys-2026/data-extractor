"""C-RET-04 -- collection guard (startup verification).

INV-RET-03 / INV-CMN-06: vectors from different models are not comparable, so a
mismatch would silently return garbage.  At startup the guard:

1. resolves each allowed alias to the real collection behind it,
2. verifies the collection has a ``dense`` vector of the model's dimension and,
   for sparse/hybrid modes, a ``sparse`` vector,
3. reads the collection's ``index_info.json`` metadata and compares its
   ``embedding_model`` with the model from the D3 registry.

Any mismatch raises :class:`CollectionGuardError` naming *both* models, and the
service refuses to start.  The alias target is not re-checked at runtime: after
an alias is repointed the service is restarted (readme 3.1.6).
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from rag_common.embeddings import spec_for
from retriever.config import RetrieverConfig
from retriever.metadata import INDEX_INFO_POINT_ID, index_info_collection


class CollectionGuardError(RuntimeError):
    """Raised when a collection is incompatible with the configured model."""


@dataclass
class CollectionStatus:
    alias: str
    resolved: str
    embedding_model: str | None
    has_sparse: bool


def _index_info(client: QdrantClient, collection: str) -> dict:
    """Read the collection's ``index_info`` metadata if present.

    Command 2 publishes per-collection metadata (notably ``embedding_model``) in a
    sibling collection ``<collection>__index_info``.  Missing metadata is not an
    error -- older collections simply have none and the vector check still runs.
    """
    meta_collection = index_info_collection(collection)
    try:
        records = client.retrieve(
            collection_name=meta_collection, ids=[INDEX_INFO_POINT_ID], with_payload=True
        )
    except Exception:
        return {}
    if not records:
        return {}
    return dict(records[0].payload or {})


def _resolve_alias(client: QdrantClient, name: str) -> str:
    """Return the concrete collection an alias points at, or ``name`` itself."""
    try:
        aliases = client.get_aliases().aliases
    except Exception:
        return name
    for alias in aliases:
        if alias.alias_name == name:
            return alias.collection_name
    return name


def check_collection(
    client: QdrantClient,
    config: RetrieverConfig,
    alias: str,
) -> CollectionStatus:
    """Verify a single collection/alias against the configured embedding model."""
    spec = spec_for(config.embedding.model)
    resolved = _resolve_alias(client, alias)

    try:
        info = client.get_collection(resolved)
    except Exception as exc:
        raise CollectionGuardError(
            f"collection/alias {alias!r} (resolved to {resolved!r}) is not available: {exc}"
        ) from exc

    vectors = info.config.params.vectors
    dense_params = vectors.get("dense") if isinstance(vectors, dict) else None
    if dense_params is None:
        raise CollectionGuardError(
            f"collection {resolved!r} has no named 'dense' vector; "
            f"expected dimension {spec.dimension} for model {spec.name!r}"
        )
    if dense_params.size != spec.dimension:
        raise CollectionGuardError(
            f"collection {resolved!r} dense dimension {dense_params.size} != "
            f"model {config.embedding.model!r} dimension {spec.dimension}"
        )

    sparse_names = set()
    if isinstance(vectors, dict):
        sparse_names = set(vectors.keys())
    has_sparse = "sparse" in sparse_names

    if config.search.mode in ("sparse", "hybrid") and not has_sparse:
        raise CollectionGuardError(
            f"collection {resolved!r} has no 'sparse' vector but search.mode="
            f"{config.search.mode!r} requires one"
        )

    meta = _index_info(client, resolved)
    index_model = meta.get("embedding_model")
    if index_model and index_model != config.embedding.model:
        raise CollectionGuardError(
            f"collection {resolved!r} was built with embedding model {index_model!r} "
            f"but the retriever is configured with {config.embedding.model!r}"
        )

    return CollectionStatus(
        alias=alias,
        resolved=resolved,
        embedding_model=index_model or config.embedding.model,
        has_sparse=has_sparse,
    )


def check_all(client: QdrantClient, config: RetrieverConfig) -> dict[str, CollectionStatus]:
    """Check every alias in ``allowed_collections`` and return their statuses."""
    return {alias: check_collection(client, config, alias) for alias in config.allowed_collections}


__all__ = ["CollectionGuardError", "CollectionStatus", "check_collection", "check_all"]

