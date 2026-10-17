"""K6 dataset loading and split handling (used by the evaluation runner)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sample:
    """One K6 row."""

    sample_id: str
    user_input: str
    history: list[dict] = field(default_factory=list)
    reference_doc_ids: list[str] = field(default_factory=list)
    reference_urls: list[str] = field(default_factory=list)
    answerable: bool = True
    split: str | None = None
    lang: str | None = None
    meta: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def load_dataset(path: str | Path) -> list[Sample]:
    """Read a K6 jsonl file into :class:`Sample` objects."""
    samples: list[Sample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        samples.append(
            Sample(
                sample_id=str(row.get("sample_id", "")),
                user_input=str(row.get("user_input", "")),
                history=list(row.get("history") or []),
                reference_doc_ids=list(row.get("reference_doc_ids") or []),
                reference_urls=list(row.get("reference_urls") or []),
                answerable=bool(row.get("answerable", True)),
                split=row.get("split"),
                lang=row.get("lang"),
                meta=dict(row.get("meta") or {}),
                raw=row,
            )
        )
    return samples


def select_split(samples: list[Sample], split: str | None) -> list[Sample]:
    """Filter samples by ``split``.

    A sample without a ``split`` field is an error (UC-RET-04 step 1): silent
    inclusion would leak validation/test rows into tuning (INV-RET-07).
    """
    if split is None:
        return list(samples)
    missing = [s.sample_id for s in samples if s.split is None]
    if missing:
        raise ValueError(
            f"{len(missing)} sample(s) have no 'split' field (e.g. {missing[0]!r}); "
            "every K6 row must declare its split (INV-RET-07)"
        )
    return [s for s in samples if s.split == split]


__all__ = ["Sample", "load_dataset", "select_split"]
