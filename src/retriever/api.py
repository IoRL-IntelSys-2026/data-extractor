"""FastAPI service: ``POST /retrieve`` and ``GET /health`` (K3, deliverable D7).

Startup (UC-RET-02): load the config, compute ``config_id``, load the embedding
model (and reranker if enabled), verify every allowed collection with the guard,
and run a warm-up query so the first real user does not pay the cold-start cost.
A collection/model mismatch aborts the process.

Qdrant connectivity failures retry a few times, then exit non-zero.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from rag_common.embeddings import get_embedder
from rag_common.schemas import (
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from retriever.config import RetrieverConfig, load_config
from retriever.core import Retriever
from retriever.guard import CollectionGuardError, CollectionStatus, check_all
from retriever.rerank import RerankerError

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("retriever.api")

load_dotenv(find_dotenv())

DEFAULT_CONFIG_PATH = os.environ.get(
    "RETRIEVER_CONFIG",
    str(Path(__file__).resolve().parent / "configs" / "retriever.yaml"),
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None


class AppState:
    config: RetrieverConfig
    retriever: Retriever
    client: QdrantClient
    statuses: dict[str, CollectionStatus]
    reranker_loaded: bool


state = AppState()


def _connect_qdrant(retries: int = 3, delay: float = 1.0) -> QdrantClient:
    """Connect to Qdrant, retrying a few times, then raise."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10.0)
            client.get_collections()
            return client
        except Exception as exc:  # pragma: no cover - needs live server
            last_exc = exc
            logger.warning("Qdrant connect attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError(f"could not connect to Qdrant at {QDRANT_URL}: {last_exc}")


def build_state(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Load config, models, verify collections, warm up (UC-RET-02)."""
    logger.info("loading config %s", config_path)
    state.config = load_config(config_path)
    logger.info("config_id = %s", state.config.config_id)

    # load embedding model (and reranker if enabled)
    state.retriever = Retriever(state.config, client=_LazyClient())

    # connect + verify collections
    client = _connect_qdrant()
    state.client = client
    state.retriever.client = client
    state.retriever._searcher.client = client

    try:
        state.statuses = check_all(client, state.config)
    except CollectionGuardError as exc:
        logger.error("collection check failed: %s", exc)
        raise

    state.reranker_loaded = state.config.rerank.enabled and state.retriever.reranker is not None

    # warm-up
    try:
        state.retriever.retrieve(RetrievalRequest(query="warmup"))
    except Exception as exc:  # pragma: no cover - warm-up best effort
        logger.warning("warm-up query failed: %s", exc)


class _LazyClient:
    """Placeholder client used before the real Qdrant connection is established."""

    def query_points(self, *args, **kwargs):  # pragma: no cover - never called
        raise RuntimeError("Qdrant client not initialised")


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_state()
    yield
    try:
        state.client.close()
    except Exception:  # pragma: no cover
        pass


app = FastAPI(title="Retriever service", version="0.1.0", lifespan=lifespan)


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    # top_k bounds (UC-RET-01 step 1)
    top_k = request.top_k if request.top_k is not None else state.config.top_k_default
    if top_k < 1 or top_k > state.config.top_k_max:
        raise HTTPException(
            status_code=422,
            detail=f"top_k must be between 1 and {state.config.top_k_max}",
        )
    # collection must be allowed
    if request.collection is not None and request.collection not in state.config.allowed_collections:
        raise HTTPException(
            status_code=422,
            detail=f"collection must be one of {state.config.allowed_collections}",
        )
    try:
        return state.retriever.retrieve(request)
    except RerankerError as exc:
        raise HTTPException(status_code=503, detail=f"reranker unavailable: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # Qdrant unavailable etc.
        logger.exception("retrieve failed")
        raise HTTPException(status_code=503, detail=f"retrieval backend unavailable: {exc}") from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        collections={alias: st.resolved for alias, st in state.statuses.items()},
        embedding_model=state.config.embedding.model,
        config_id=state.config.config_id,
        reranker_loaded=state.reranker_loaded,
    )


@app.exception_handler(CollectionGuardError)
async def _guard_handler(request: Request, exc: CollectionGuardError):  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("RETRIEVER_PORT", "8001")), workers=1)


if __name__ == "__main__":  # pragma: no cover
    main()
