"""C-RET-02 -- synonym / abbreviation dictionary for RUDN.

Rules (readme 3.1.5 and T-RET-20):

* terms are expanded on **word boundaries** -- ``общаговский`` must not match
  ``общага``,
* the original word is *not* replaced; the full form is *appended*, so
  ``"общага"`` becomes ``"общага общежитие"``,
* ambiguous entries (``ambiguous: true``, e.g. ``ПК`` = "приёмная комиссия" vs
  "повышение квалификации") are only expanded when one of the ``context`` words
  is present in the query.

The dictionary version participates in the config hash, so any change is visible
in ``config_id`` (INV-RET-04).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Entry:
    term: str
    expand: tuple[str, ...]
    type: str = "synonym"
    ambiguous: bool = False
    context: tuple[str, ...] = field(default_factory=tuple)
    source: str | None = None


def _context_matches(context_word: str, lowered_query: str) -> bool:
    """Match a context word allowing simple inflections.

    Context words in the dictionary are written in their base form
    (``поступление``) but real queries use inflections (``поступления``).  We
    match on a word-start *stem*: the base form with its trailing character
    dropped (the typical inflection slot), requiring a minimum stem length so we
    do not over-match short words.
    """
    word = context_word.lower()
    stem = word[:-1] if len(word) > 4 else word
    escaped = re.escape(stem)
    return re.search(rf"(?<!\w){escaped}\w*", lowered_query) is not None


class Dictionary:
    """Loaded synonym dictionary with word-boundary expansion."""

    def __init__(self, entries: list[Entry], *, version: int = 1) -> None:
        self.entries = entries
        self.version = version

    # -- loading -------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Dictionary":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = [
            Entry(
                term=str(item["term"]),
                expand=tuple(str(x) for x in item.get("expand", [])),
                type=str(item.get("type", "synonym")),
                ambiguous=bool(item.get("ambiguous", False)),
                context=tuple(str(x) for x in item.get("context", [])),
                source=item.get("source"),
            )
            for item in raw.get("entries", [])
            if "term" in item
        ]
        cls._validate(entries)
        return cls(entries, version=int(raw.get("version", 1)))

    @staticmethod
    def _validate(entries: list[Entry]) -> None:
        """Validate structure and duplicates.

        A term may legitimately appear more than once *only* as competing
        ``ambiguous`` interpretations (e.g. ``ПК`` = "приёмная комиссия" vs
        "повышение квалификации"), each with its own ``context``.  A plain
        duplicate (non-ambiguous) is an error.
        """
        by_term: dict[str, list[Entry]] = {}
        for entry in entries:
            if not entry.expand:
                raise ValueError(f"dictionary entry {entry.term!r} has no expansion")
            if not entry.type:
                raise ValueError(f"dictionary entry {entry.term!r} has no type")
            if entry.ambiguous and not entry.context:
                raise ValueError(
                    f"ambiguous entry {entry.term!r} must declare context words"
                )
            by_term.setdefault(entry.term.lower(), []).append(entry)

        for key, group in by_term.items():
            if len(group) == 1:
                continue
            if not all(e.ambiguous for e in group):
                raise ValueError(
                    f"duplicate dictionary term {group[0].term!r}; "
                    "only ambiguous entries may repeat a term"
                )
            contexts = [tuple(e.context) for e in group]
            if len(set(contexts)) != len(contexts):
                raise ValueError(
                    f"ambiguous entries for {group[0].term!r} reuse the same context"
                )

    # -- expansion -----------------------------------------------------------

    def expand(self, query: str) -> str:
        """Return the extra text to append to ``query`` (possibly empty)."""
        lowered = query.lower()
        added: list[str] = []
        for entry in self.entries:
            pattern = re.compile(rf"(?<!\w){re.escape(entry.term.lower())}(?!\w)")
            if not pattern.search(lowered):
                continue
            if entry.ambiguous and not any(
                _context_matches(ctx, lowered) for ctx in entry.context
            ):
                continue
            added.extend(entry.expand)
        # de-duplicate while keeping order
        seen: set[str] = set()
        unique: list[str] = []
        for token in added:
            low = token.lower()
            if low not in seen:
                seen.add(low)
                unique.append(token)
        return " ".join(unique)


def load_dictionary(path: str | Path | None) -> Dictionary | None:
    """Load a dictionary from ``path`` or return ``None`` when not configured."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return Dictionary.from_yaml(p)


__all__ = ["Dictionary", "Entry", "load_dictionary"]


