"""Light-weight, self-contained subset of the shared ``rag_common`` package.

In the full project this package is owned by the platform teams (D1/D2/D3).  The
retriever only *consumes* a few contracts from it:

* :mod:`rag_common.schemas`  -- K3 request/response models (RET owns the K3 part),
* :mod:`rag_common.embeddings` -- the query embedding module (D3),
* :mod:`rag_common.metrics` -- retrieval metrics (D6).

This module ships standalone implementations so the retriever can be built, run
and tested without pulling in the heavy sibling repositories.  The public
interfaces (``get_embedder().encode_query()`` etc.) match the contracts described
in the readme so the real package can be dropped in unchanged.
"""

__all__ = ["schemas", "embeddings", "metrics"]
