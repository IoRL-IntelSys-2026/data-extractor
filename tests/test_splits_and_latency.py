"""T-RET-07 (honest splits) and T-RET-17 (latency_stats)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retriever.eval.dataset import load_dataset, select_split
from retriever.eval.runner import run_eval
from retriever.tools.latency import latency_stats

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "k6"


# -- T-RET-07 ---------------------------------------------------------------

def test_select_split_filters():
    samples = load_dataset(FIXTURES / "k6_sample.jsonl")
    dev = select_split(samples, "dev")
    assert {s.split for s in dev} == {"dev"}
    assert len(dev) == 1


def test_missing_split_is_error(tmp_path):
    path = tmp_path / "nosp.jsonl"
    path.write_text(json.dumps({"sample_id": "x", "user_input": "q"}) + "\n", encoding="utf-8")
    samples = load_dataset(path)
    with pytest.raises(ValueError):
        select_split(samples, "train")


def test_tune_on_validation_rejected(make_config, fake_embedder, tmp_path):
    from tests.conftest import FakeQdrantClient

    cfg = make_config()
    retriever = _build(fake_embedder, cfg)
    with pytest.raises(ValueError):
        run_eval(
            config_path=_write_config(tmp_path, "dense.yaml"),
            dataset_path=FIXTURES / "k6_sample.jsonl",
            split="validation",
            variant="last",
            retriever=retriever,
            output_root=tmp_path / "runs",
            tune=True,
        )


def test_test_dataset_requires_final(make_config, fake_embedder, tmp_path):
    cfg = make_config()
    retriever = _build(fake_embedder, cfg)
    with pytest.raises(ValueError):
        run_eval(
            config_path=_write_config(tmp_path, "dense.yaml"),
            dataset_path=FIXTURES / "mdd_eval_test_v1.jsonl",
            split="test",
            variant="last",
            retriever=retriever,
            output_root=tmp_path / "runs",
            final=False,
        )


def test_eval_writes_run_dir_and_report(make_config, fake_embedder, tmp_path, monkeypatch):
    import retriever.eval.runner as runner

    monkeypatch.setattr(runner, "REPORT_PATH", tmp_path / "ret_runs.md")
    cfg = make_config()
    retriever = _build(fake_embedder, cfg)
    summary = run_eval(
        config_path=_write_config(tmp_path, "dense.yaml"),
        dataset_path=FIXTURES / "k6_sample.jsonl",
        split="dev",
        variant="last",
        retriever=retriever,
        output_root=tmp_path / "runs",
    )
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "samples.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    config_json = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_json["config_id"] == cfg.config_id
    assert config_json["generation_model"] is None
    assert config_json["judge_model"] is None
    assert (tmp_path / "ret_runs.md").exists()


# -- T-RET-17 ---------------------------------------------------------------

def test_latency_stats_basic(tmp_path):
    path = tmp_path / "samples.jsonl"
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    with path.open("w", encoding="utf-8") as fh:
        for v in values:
            fh.write(json.dumps({"latency_ms": v}) + "\n")
    stats = latency_stats(path)
    assert stats["n"] == 10
    assert stats["p50"] == 55.0
    assert stats["p95"] >= 90


def test_latency_skips_bad_rows(tmp_path):
    path = tmp_path / "samples.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"latency_ms": 10}) + "\n")
        fh.write(json.dumps({"latency_ms": None}) + "\n")
        fh.write(json.dumps({"latency_ms": "oops"}) + "\n")
        fh.write("not json\n")
        fh.write(json.dumps({"no_latency": 1}) + "\n")
    stats = latency_stats(path)
    assert stats["n"] == 1


def test_latency_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    stats = latency_stats(path)
    assert stats["n"] == 0
    assert stats["p50"] is None


# -- helpers ----------------------------------------------------------------

def _build(fake_embedder, cfg):
    from retriever.core import Retriever
    from tests.conftest import FakeQdrantClient

    return Retriever(cfg, FakeQdrantClient(), embedder=fake_embedder)


def _write_config(tmp_path, name):
    import yaml

    data = {
        "collection": "mdd_current",
        "allowed_collections": ["mdd_current", "rudn_current"],
        "embedding": {"model": "BAAI/bge-m3"},
        "query": {"max_tokens": 256, "dictionary": {"enabled": False}},
        "search": {"mode": "dense", "prefetch_limit": 50, "candidates": 40},
        "rerank": {"enabled": False},
        "postprocess": {"max_per_doc": 2},
        "top_k_default": 5,
        "top_k_max": 20,
        "no_relevant": {"signal": "cosine", "threshold": None},
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path
