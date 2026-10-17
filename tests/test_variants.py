"""T-RET-19 -- query variants for the evaluation runner."""

from __future__ import annotations

from retriever.eval.dataset import Sample
from retriever.eval.variants import build_concat, build_variant


def _sample(user_input, history, lang="en"):
    return Sample(sample_id="s1", user_input=user_input, history=history, lang=lang)


def test_first_turn_all_variants_equal_user_input():
    sample = _sample("Hello?", history=[])
    for variant in ("last", "concat"):
        assert build_variant(sample, variant).text == "Hello?"


def test_concat_reverse_order_and_budget():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    sample = _sample("third", history=history)
    text = build_concat(sample)
    assert text.startswith("third")
    assert "second" in text and "first" in text


def test_concat_truncates_to_budget():
    history = [{"role": "user", "content": " ".join(["w"] * 300)}]
    sample = _sample("hello", history=history)
    text = build_concat(sample, budget=128)
    assert len(text.split(" ")) <= 128
