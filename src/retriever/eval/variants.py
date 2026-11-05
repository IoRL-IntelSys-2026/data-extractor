"""C-RET-03 -- query variants for dialogue evaluation.

Three variants (UC-RET-03):

* ``last``   -- only the latest user turn (``user_input``),
* ``concat`` -- current turn + previous turns in reverse order, truncated to a
  token budget (start 128, as in the multidoc2dial paper),
* ``rewrite`` -- the standalone question produced by ``rag_common.rewrite`` (the
  same module and prompt ``/answer`` uses), imported *only* here (INV-RET-05).

On the first turn there is no history, so all three variants equal ``user_input``.
"""

from __future__ import annotations

from dataclasses import dataclass

from retriever.eval.dataset import Sample

CONCAT_TOKEN_BUDGET = 128


@dataclass
class VariantQuery:
    variant: str
    text: str
    rewrite_fallback: bool = False
    prompt_version: str | None = None
    rewrite_model: str | None = None


def _tokenize(text: str) -> list[str]:
    return text.split()


def build_concat(sample: Sample, budget: int = CONCAT_TOKEN_BUDGET) -> str:
    """Current turn + earlier turns in reverse order, truncated to ``budget`` tokens.

    ``history`` is assumed to be chronological (oldest first); we walk it
    backwards so the most recent context is kept when truncating.
    """
    turns = [t for t in sample.history if t.get("role") == "user"]
    pieces = [sample.user_input]
    for turn in reversed(turns):
        pieces.append(str(turn.get("content", "")))
    joined = " ".join(p for p in pieces if p)
    tokens = _tokenize(joined)
    if len(tokens) > budget:
        joined = " ".join(tokens[:budget])
    return joined.strip()


def build_variant(sample: Sample, variant: str) -> VariantQuery:
    """Build the query text for a named variant."""
    if variant == "last":
        return VariantQuery(variant="last", text=sample.user_input)
    if variant == "concat":
        return VariantQuery(variant="concat", text=build_concat(sample))
    if variant == "rewrite":
        from rag_common import rewrite as rewrite_mod

        result = rewrite_mod.rewrite(
            sample.user_input,
            sample.history,
            sample.lang or "ru",
        )
        return VariantQuery(
            variant="rewrite",
            text=result.text,
            rewrite_fallback=result.fallback,
            prompt_version=result.prompt_version,
            rewrite_model=result.model,
        )
    raise ValueError(f"unknown query variant: {variant!r}")


__all__ = ["VariantQuery", "build_variant", "build_concat", "CONCAT_TOKEN_BUDGET"]
