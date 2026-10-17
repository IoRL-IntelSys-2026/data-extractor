"""T-RET-08 (no_relevant rule), T-RET-10 (max_per_doc) and reranker wiring."""

from __future__ import annotations

import pytest

from retriever.config import NoRelevantConfig, PostprocessConfig
from retriever.postprocess import apply_max_per_doc, compute_no_relevant, postprocess
from retriever.rerank import RerankerError, rerank_candidates
from retriever.search import Candidate


def _cand(chunk_id, doc_id, score, **payload):
    base = {"chunk_id": chunk_id, "doc_id": doc_id, "url": "u", "title": "t", "text": "x"}
    base.update(payload)
    return Candidate(chunk_id=chunk_id, score=score, payload=base)


# -- T-RET-10 ---------------------------------------------------------------

def test_max_per_doc():
    cands = [_cand(f"c{i}", "d1" if i < 6 else "d2", 1.0 - i * 0.01) for i in range(10)]
    kept = apply_max_per_doc(cands, 2)
    from collections import Counter

    counts = Counter(c.payload["doc_id"] for c in kept)
    assert counts["d1"] == 2
    assert counts["d2"] == 2
    assert len(kept) == 4


def test_max_per_doc_returns_fewer_but_never_violates():
    # 6 chunks all from one doc, limit 2, top_k 5 -> only 2 returned
    cands = [_cand(f"c{i}", "d1", 1.0 - i * 0.01) for i in range(6)]
    result = postprocess(
        cands,
        postprocess_config=PostprocessConfig(max_per_doc=2),
        no_relevant_config=NoRelevantConfig(signal="cosine", threshold=None),
        top_k=5,
    )
    assert len(result.chunks) == 2


# -- T-RET-08 ---------------------------------------------------------------

def test_no_relevant_rule():
    cfg = NoRelevantConfig(signal="rerank_score", threshold=0.15)
    # above threshold
    ok = [_cand("a", "d", 0.9)]
    ok[0].rerank_score = 0.9
    below = [_cand("a", "d", 0.05)]
    below[0].rerank_score = 0.05
    assert compute_no_relevant(ok, cfg)[0] is False
    assert compute_no_relevant(below, cfg)[0] is True
    assert compute_no_relevant([], cfg)[0] is True


def test_no_relevant_rule_disabled_with_null_threshold():
    cfg = NoRelevantConfig(signal="cosine", threshold=None)
    cand = [_cand("a", "d", 0.0)]
    assert compute_no_relevant(cand, cfg)[0] is False
    assert compute_no_relevant([], cfg)[0] is True


def test_no_relevant_chunks_returned_anyway():
    cfg = NoRelevantConfig(signal="cosine", threshold=0.99)
    result = postprocess(
        [_cand("a", "d", 0.1)],
        postprocess_config=PostprocessConfig(),
        no_relevant_config=cfg,
        top_k=5,
    )
    assert result.no_relevant is True
    assert len(result.chunks) == 1  # still returned for debugging/eval


# -- reranker wiring --------------------------------------------------------

class _ConstReranker:
    def __init__(self, scores):
        self._scores = scores

    def score(self, query, candidates):
        return self._scores[: len(candidates)]


def test_reranker_reorders():
    cands = [_cand("a", "d1", 0.9), _cand("b", "d2", 0.8)]
    rr = _ConstReranker([0.1, 0.95])
    out = rerank_candidates(rr, "q", cands)
    assert [c.chunk_id for c in out] == ["b", "a"]
    assert out[0].rerank_score == 0.95


def test_reranker_failure_raises():
    class _Boom:
        def score(self, query, candidates):
            raise RuntimeError("gpu oom")

    with pytest.raises(RerankerError):
        rerank_candidates(_Boom(), "q", [_cand("a", "d", 0.5)])


def test_reranker_mismatched_count_raises():
    with pytest.raises(RerankerError):
        rerank_candidates(_ConstReranker([0.1]), "q", [_cand("a", "d", 0.5), _cand("b", "d", 0.4)])
