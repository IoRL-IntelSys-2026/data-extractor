"""C-RET-09 helper -- p50/p95 latency from a run's ``samples.jsonl`` (or a log).

T-RET-17: reads the ``latency_ms`` field, skips empty and non-numeric values with
a warning, and returns p50 and p95.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger("retriever.latency")


def _iter_latency_ms(path: str | Path):
    path = Path(path)
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("%s:%d not JSON, skipped", path, lineno)
            continue
        value = row.get("latency_ms")
        if value is None or value == "":
            logger.warning("%s:%d missing latency_ms, skipped", path, lineno)
            continue
        try:
            yield float(value)
        except (TypeError, ValueError):
            logger.warning("%s:%d non-numeric latency_ms %r, skipped", path, lineno, value)


def latency_stats(path: str | Path) -> dict[str, float | int | None]:
    """Return ``{"p50": .., "p95": .., "n": ..}`` for a ``samples.jsonl`` file."""
    values = list(_iter_latency_ms(path))
    if not values:
        return {"p50": None, "p95": None, "n": 0}
    values.sort()
    p50 = statistics.median(values)
    p95 = _percentile(values, 95)
    return {"p50": p50, "p95": p95, "n": len(values)}


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile (matches the common small-sample convention)."""
    if not sorted_values:
        raise ValueError("empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


__all__ = ["latency_stats"]
