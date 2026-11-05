"""C-RET-10 -- load the static K2 fixture into a test collection.

This is a *test* helper, not a deliverable.  It loads the K2 fixture points into
``mdd_exp_test_v1`` in the isolated test Qdrant, writes the ``index_info``
metadata (``embedding_model``) into a sibling ``__index_info`` collection so the
guard can verify the model, and can drop everything afterwards.  It never touches
production aliases.
"""

from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient, models

from retriever.metadata import INDEX_INFO_POINT_ID, index_info_collection

FIXTURE_DIR = Path("tests/fixtures/k2")


def load_fixture(
    client: QdrantClient,
    *,
    collection: str = "mdd_exp_test_v1",
    fixture_dir: str | Path = FIXTURE_DIR,
    embedding_model: str = "BAAI/bge-m3",
    recreate: bool = True,
) -> None:
    """Create ``collection`` and load dense (+sparse) fixture vectors."""
    fixture_dir = Path(fixture_dir)
    points = [json.loads(line) for line in (fixture_dir / "points.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    has_sparse = any(p.get("sparse") for p in points)

    if recreate:
        client.delete_collection(collection)
    vectors_config = {"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)}
    sparse_config = {"sparse": models.SparseVectorParams()} if has_sparse else {}
    if recreate or not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config or None,
        )

    qdrant_points = []
    for p in points:
        vector: dict = {"dense": p["dense"]}
        if has_sparse and p.get("sparse"):
            vector["sparse"] = models.SparseVector(
                indices=p["sparse"]["indices"], values=p["sparse"]["values"]
            )
        qdrant_points.append(models.PointStruct(id=p["id"], vector=vector, payload=p["payload"]))
    client.upsert(collection_name=collection, points=qdrant_points)

    _write_index_info(client, collection, embedding_model, has_sparse)


def _write_index_info(
    client: QdrantClient, collection: str, embedding_model: str, has_sparse: bool
) -> None:
    """Write metadata into the sibling ``<collection>__index_info`` collection."""
    meta_collection = index_info_collection(collection)
    try:
        client.delete_collection(meta_collection)
    except Exception:
        pass
    client.create_collection(
        collection_name=meta_collection,
        vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=meta_collection,
        points=[
            models.PointStruct(
                id=INDEX_INFO_POINT_ID,
                vector=[0.0],
                payload={
                    "embedding_model": embedding_model,
                    "has_sparse": has_sparse,
                    "source_collection": collection,
                },
            )
        ],
    )


def drop_collection(client: QdrantClient, collection: str = "mdd_exp_test_v1") -> None:
    for name in (collection, index_info_collection(collection)):
        try:
            client.delete_collection(name)
        except Exception:
            pass


__all__ = ["load_fixture", "drop_collection"]

