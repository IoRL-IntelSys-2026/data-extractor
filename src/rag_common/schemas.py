"""K3 request/response schemas.

The readme (section 3.1.3, INV-RET-01 / INV-RET-06 and UC-RET-01) fixes the shape
of the retriever's HTTP contract.  The rules K3 from ``0.6`` are not changed here,
only made concrete.

Key constraints enforced by these models:

* the query is a *standalone* question -- there is no ``history`` field and a
  request carrying one is rejected with 422 (INV-RET-05),
* ``top_k`` must be within ``1..top_k_max``,
* filter field *names* are validated (only ``lang, section, source, content_type,
  doc_id`` are known); filter *values* are not validated because an unknown
  ``section`` value should legitimately return an empty result set,
* a blank query is rejected with 422.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FilterSpec(BaseModel):
    """Filter fields accepted by ``/retrieve`` (INV-RET-06).

    Every filter value is either a single string or a list of strings and is
    matched with "any of" semantics.
    """

    model_config = ConfigDict(extra="forbid")

    lang: str | list[str] | None = None
    section: str | list[str] | None = None
    source: str | list[str] | None = None
    content_type: str | list[str] | None = None
    doc_id: str | list[str] | None = None

    def as_map(self) -> dict[str, list[str]]:
        """Return only the set filters, normalised to lists of strings."""
        out: dict[str, list[str]] = {}
        for key in ("lang", "section", "source", "content_type", "doc_id"):
            value = getattr(self, key)
            if value is None:
                continue
            out[key] = [value] if isinstance(value, str) else list(value)
        return out


class RetrievalRequest(BaseModel):
    """POST /retrieve request body (K3).

    ``extra="forbid"`` is what turns a stray ``history`` field into a 422
    (INV-RET-05) -- the retriever never rewrites questions, that is command 4's
    job before the call.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int | None = None
    filters: FilterSpec | None = None
    collection: str | None = None
    debug: bool = False

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if value is None or not value.strip():
            raise ValueError("query must not be empty")
        return value


class RetrievedChunk(BaseModel):
    """A single retrieved chunk (K3 chunk object)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    url: str
    title: str
    text: str
    score: float
    rank: int
    payload: dict[str, Any] | None = None


class DebugInfo(BaseModel):
    """Optional ``debug`` payload (only present when ``debug = true``)."""

    model_config = ConfigDict(extra="allow")

    timing_ms: dict[str, float] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    relevance_signal: float | None = None
    resolved_collection: str | None = None


class RetrievalResponse(BaseModel):
    """POST /retrieve response body (K3)."""

    model_config = ConfigDict(extra="forbid")

    search_query: str
    no_relevant: bool
    collection: str
    config_id: str
    latency_ms: float
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    debug: DebugInfo | None = None

    @field_validator("search_query")
    @classmethod
    def _search_query_not_null(cls, value: str) -> str:
        # INV-RET-01: search_query is never null.
        if value is None:
            raise ValueError("search_query must not be null")
        return value


class HealthResponse(BaseModel):
    """GET /health response body."""

    model_config = ConfigDict(extra="allow")

    status: Literal["ok", "degraded"]
    collections: dict[str, str] = Field(default_factory=dict)
    embedding_model: str
    config_id: str
    reranker_loaded: bool


__all__ = [
    "FilterSpec",
    "RetrievalRequest",
    "RetrievedChunk",
    "DebugInfo",
    "RetrievalResponse",
    "HealthResponse",
]
