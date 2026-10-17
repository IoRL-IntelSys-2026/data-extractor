"""C-RET-01 -- configuration model and automatic ``config_id``.

Responsibilities (readme 3.2.1):

* validate the YAML profile with pydantic,
* require the RRF ``k`` explicitly (INV-RET-02): Qdrant's default is ``k = 2``,
  not the "textbook" 60, and results would silently differ,
* validate ``top_k`` bounds and the allowed search modes,
* compute a stable hash of the *normalised* config (key order and comments do not
  affect it) and derive ``config_id = <profile name>@<first 8 hash chars>``
  (INV-RET-04).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SearchMode = Literal["dense", "sparse", "hybrid"]


class DictionaryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str = "dictionaries/rudn_synonyms.yaml"


class QueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=256, ge=1)
    dictionary: DictionaryConfig = Field(default_factory=DictionaryConfig)


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["rrf"] = "rrf"
    # k must be explicit (INV-RET-02).  No default on purpose so a missing k
    # fails validation instead of silently becoming Qdrant's k = 2.
    k: int = Field(ge=1)


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: SearchMode = "dense"
    prefetch_limit: int = Field(default=50, ge=1)
    candidates: int = Field(default=40, ge=1)
    fusion: FusionConfig | None = None

    @model_validator(mode="after")
    def _hybrid_needs_fusion(self) -> "SearchConfig":
        if self.mode == "hybrid" and self.fusion is None:
            raise ValueError("hybrid search requires an explicit fusion.k (INV-RET-02)")
        return self


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "BAAI/bge-m3"


class RerankConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = Field(default=512, ge=1)
    batch_size: int = Field(default=16, ge=1)


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_per_doc: int | None = Field(default=None, ge=1)


class NoRelevantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: Literal["rerank_score", "cosine", "rrf", "dot"] = "rrf"
    # ``null`` disables the rule: no_relevant is then only true with 0 candidates
    # (INV-RET-09).  Uncalibrated profiles keep it null.
    threshold: float | None = None


class RetrieverConfig(BaseModel):
    """Validated retriever profile."""

    model_config = ConfigDict(extra="forbid")

    collection: str
    allowed_collections: list[str]
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    top_k_default: int = Field(default=5, ge=1)
    top_k_max: int = Field(default=20, ge=1)
    no_relevant: NoRelevantConfig = Field(default_factory=NoRelevantConfig)

    # Not part of the user YAML; injected when loading a profile so that
    # ``config_id`` can be prefixed with the profile name.
    profile_name: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_collections_and_topk(self) -> "RetrieverConfig":
        if self.collection not in self.allowed_collections:
            raise ValueError(
                f"collection {self.collection!r} must be listed in allowed_collections"
            )
        if self.top_k_default > self.top_k_max:
            raise ValueError("top_k_default must not exceed top_k_max")
        return self

    # -- config_id -----------------------------------------------------------

    def normalized_hash(self) -> str:
        """Stable 8-char hash of the normalised config (INV-RET-04).

        The profile name is excluded so that two profiles with the same body but
        different names still hash to the same digest *body*; the name is added
        separately in :meth:`config_id`.  ``json.dumps`` with sorted keys makes
        the digest independent of YAML key order and comments.
        """
        data = self.model_dump(exclude={"profile_name"})
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

    @property
    def config_id(self) -> str:
        """``<profile name>@<first 8 hash chars>``, e.g. ``dense@1f3a9c2e``."""
        name = self.profile_name or _slugify(self.collection)
        return f"{name}@{self.normalized_hash()}"


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_")


def load_config(path: str | Path, *, profile_name: str | None = None) -> RetrieverConfig:
    """Load, validate and annotate a YAML profile.

    When ``profile_name`` is omitted it is derived from the file stem, which per
    the readme is the first part of ``config_id``.
    """
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if profile_name is None:
        profile_name = path.stem
    raw["profile_name"] = profile_name
    return RetrieverConfig.model_validate(raw)


__all__ = [
    "RetrieverConfig",
    "SearchConfig",
    "FusionConfig",
    "EmbeddingConfig",
    "RerankConfig",
    "PostprocessConfig",
    "NoRelevantConfig",
    "DictionaryConfig",
    "QueryConfig",
    "load_config",
]
