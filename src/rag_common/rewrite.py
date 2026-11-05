"""Question rewriting with dialogue history (K5, owned by command 4).

The retriever itself never calls this (INV-RET-05).  Only the *evaluation runner*
uses it, and only for the ``rewrite`` variant, so it lives in ``rag_common`` and
is imported exclusively under ``retriever/eval/``.

The real implementation calls an LLM with command 4's prompt.  This shim exposes
the same interface and raises :class:`RewriteUnavailable` when no LLM is
configured -- which is exactly what the evaluation runner must surface as a
clear error instead of silently substituting the ``concat`` variant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class RewriteUnavailable(RuntimeError):
    """Raised when the rewriting model is not reachable in an evaluation run."""


@dataclass
class RewriteResult:
    text: str
    fallback: bool = False
    prompt_version: str = "v1"
    model: str = "unknown"


def rewrite(question: str, history: list[dict], lang: str = "ru") -> RewriteResult:
    """Rewrite a clarification question into a standalone one.

    ``history`` is a list of ``{"role", "content"}`` dicts.  Without a configured
    LLM this raises :class:`RewriteUnavailable` so an evaluation run fails loudly
    rather than silently producing a different variant.
    """
    if not os.environ.get("LLM_API_KEY") and not os.environ.get("LLM_BASE_URL"):
        raise RewriteUnavailable(
            "rag_common.rewrite requires a configured LLM (set LLM_BASE_URL/LLM_API_KEY); "
            "the rewrite variant cannot run without it"
        )
    # Real implementation would call the LLM here.  Kept as a stub so the
    # interface and error semantics are testable.
    raise RewriteUnavailable("LLM rewriting backend not implemented in this build")  # pragma: no cover


__all__ = ["rewrite", "RewriteResult", "RewriteUnavailable"]
