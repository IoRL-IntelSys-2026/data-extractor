"""C-RET-09 -- offline evaluation runner (``ret-eval``).

UC-RET-04: read a K6 dataset, select rows by ``split``, build the variant query,
call the :class:`Retriever` library directly, compute D6 metrics (document level)
with breakdowns, and write the run directory + a row in ``ret_runs.md``.

Guarantees honest evaluation (INV-RET-07):

* ``--tune`` is rejected on validation / test / holdout,
* the ``mdd_eval_test_v1`` file is only accepted with ``--final``,
* rows without a ``split`` are an error.

Each run writes ``config.json`` with the git commit, ``config_id``, full YAML,
requested collection and the concrete collection behind the alias, the dataset
and split, variant, and model names (INV-RET-10).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rag_common.metrics import hit_at_k, mrr_at_k, recall_at_k
from rag_common.schemas import RetrievalRequest
from retriever.config import load_config
from retriever.core import Retriever
from retriever.eval.dataset import Sample, load_dataset, select_split
from retriever.eval.variants import build_variant
from retriever.tools.latency import latency_stats

PROTECTED_SPLITS = {"validation", "test", "holdout"}
REPORT_PATH = Path("docs/reports/ret_runs.md")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _is_final_dataset(dataset: str | Path) -> bool:
    return "test" in Path(dataset).stem


def run_eval(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    split: str | None,
    variant: str,
    retriever: Retriever | None = None,
    client=None,
    output_root: str | Path = "data/eval/runs",
    tune: bool = False,
    final: bool = False,
    max_workers: int = 4,
) -> dict:
    """Run an evaluation and return the summary dict."""
    config = load_config(config_path)

    if tune and split in PROTECTED_SPLITS:
        raise ValueError(f"--tune is not allowed on split={split!r} (INV-RET-07)")
    if _is_final_dataset(dataset_path) and not final:
        raise ValueError(
            f"{Path(dataset_path).name} must only be run with --final (INV-RET-07)"
        )

    samples = load_dataset(dataset_path)
    selected = select_split(samples, split)

    if retriever is None:
        from qdrant_client import QdrantClient  # local import to keep tests light

        client = client or QdrantClient(url="http://localhost:6333")
        retriever = Retriever(config, client)

    # build variant queries (the rewrite variant may touch rag_common.rewrite)
    prepared: list[tuple[Sample, object]] = []
    for sample in selected:
        prepared.append((sample, build_variant(sample, variant)))

    def _one(item):
        sample, vq = item
        resp = retriever.retrieve(
            RetrievalRequest(query=vq.text, top_k=config.top_k_default, debug=False)
        )
        return sample, vq, resp

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, prepared))

    # assemble samples.jsonl rows and metrics
    rows: list[dict] = []
    hits: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    by_domain: dict[str, list[float]] = {}
    by_turn: dict[str, list[float]] = {}
    missing_in_index = 0

    for sample, vq, resp in results:
        retrieved_ids = [c.chunk_id for c in resp.chunks]
        retrieved_docs = [c.doc_id for c in resp.chunks]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "variant": vq.variant,
                "search_text": resp.search_query,
                "retrieved_context_ids": retrieved_ids,
                "retrieved_doc_ids": retrieved_docs,
                "scores": [c.score for c in resp.chunks],
                "reference_doc_ids": sample.reference_doc_ids,
                "no_relevant": resp.no_relevant,
                "rewrite_fallback": getattr(vq, "rewrite_fallback", False),
                "latency_ms": resp.latency_ms,
                "split": sample.split,
                "meta": sample.meta,
            }
        )
        if not sample.answerable:
            continue
        hit = hit_at_k(retrieved_docs, sample.reference_doc_ids, k=5)
        hits.append(hit)
        mrrs.append(mrr_at_k(retrieved_docs, sample.reference_doc_ids, k=10))
        recalls.append(recall_at_k(retrieved_docs, sample.reference_doc_ids, k=5))
        domain = str(sample.meta.get("domain", "unknown"))
        by_domain.setdefault(domain, []).append(hit)
        turn = "first" if sample.meta.get("is_first_turn") else "follow"
        by_turn.setdefault(turn, []).append(hit)

    summary = {
        "config_id": config.config_id,
        "dataset": str(dataset_path),
        "split": split,
        "variant": variant,
        "n": len(rows),
        "eval_n": len(hits),
        "hit_at_5": _mean(hits),
        "mrr_at_10": _mean(mrrs),
        "recall_at_5": _mean(recalls),
        "missing_in_index": missing_in_index,
        "by_domain": {k: _mean(v) for k, v in by_domain.items()},
        "by_turn": {k: _mean(v) for k, v in by_turn.items()},
    }

    # write run directory (K6 structure)
    run_dir = _run_dir(output_root, config.config_id, dataset_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["latency"] = latency_stats(run_dir / "samples.jsonl")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # config.json (INV-RET-10)
    config_dump = {
        "git_commit": _git_commit(),
        "config_id": config.config_id,
        "profile": config_path.name if hasattr(config_path, "name") else str(config_path),
        "yaml": yaml.safe_load(Path(config_path).read_text(encoding="utf-8")),
        "requested_collection": config.collection,
        "resolved_collection": None,  # filled by the guard at service start
        "dataset": str(dataset_path),
        "split": split,
        "variant": variant,
        "embedding_model": config.embedding.model,
        "reranker_model": config.rerank.model if config.rerank.enabled else None,
        "rewrite_prompt_version": None,
        "rewrite_model": None,
        # K6 generation / judge fields are intentionally null for retrieval runs
        "generation_model": None,
        "judge_model": None,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _append_report_row(run_dir, summary)
    summary["run_dir"] = str(run_dir)
    return summary


def _run_dir(root: str | Path, config_id: str, dataset_path: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dataset_name = Path(dataset_path).stem
    return Path(root) / f"ret-{stamp}-{config_id}-{dataset_name}"


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _append_report_row(run_dir: Path, summary: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| run | config_id | dataset | split | variant | Hit@5 | MRR@10 | p95 ms |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    if not REPORT_PATH.exists():
        REPORT_PATH.write_text("# Retriever runs\n\n" + header, encoding="utf-8")
    latency = summary.get("latency") or {}
    p95 = latency.get("p95")
    row = (
        f"| {run_dir.name} | {summary['config_id']} | {Path(summary['dataset']).name} | "
        f"{summary['split']} | {summary['variant']} | {summary['hit_at_5']} | "
        f"{summary['mrr_at_10']} | {p95} |\n"
    )
    with REPORT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(row)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(prog="ret-eval", description="Offline retriever evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--variant", default="last", choices=["last", "concat", "rewrite"])
    parser.add_argument("--output-root", default="data/eval/runs")
    parser.add_argument("--tune", action="store_true", help="tuning mode (train/dev only)")
    parser.add_argument("--final", action="store_true", help="allow the test dataset")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    summary = run_eval(
        config_path=args.config,
        dataset_path=args.dataset,
        split=args.split,
        variant=args.variant,
        output_root=args.output_root,
        tune=args.tune,
        final=args.final,
        max_workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
